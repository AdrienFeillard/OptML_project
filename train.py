import os
import time
import datetime
import typer
import torch
#import torch_directml
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.traceback import install as install_rich_traceback

# Import from reorganized modules
from core.data import CIFAR10Data
from core.module import CIFAR10Module
from utils.logging import LogHandler
from utils.metrics import TrainingMetrics
from utils.visualization import create_dashboard, visualize_lr_schedule
from utils.configs.config import Classifier, LoggerType, NoiseDistribution
from core.noise_regularization import NoiseType, NoiseSchedule
# Install rich traceback handler
install_rich_traceback(show_locals=True)

# Initialize console
console = Console()

def main(
        download_weights: bool = typer.Option(False, "--download-weights", "-w", help="Download pre-trained weights"),

        # Data & model configuration
        data_dir: str = "./data/cifar10",
        classifier: Classifier = typer.Option(Classifier.resnet18, "--classifier", "-c", help="Classifier model to use"),

        # Training parameters
        batch_size: int = typer.Option(128, "--batch-size", "-b", help="Batch size for training"),
        max_epochs: int = typer.Option(100, "--epochs", "-e", help="Maximum number of epochs"),
        num_workers: int = typer.Option(4, "--workers", "-n", help="Number of data loading workers"),
        subset: float = typer.Option(1.0, "--subset", "-s", help="Fraction of dataset to use for training (0.0-1.0)"),

        # Optimizer configuration
        learning_rate: float = typer.Option(1e-2, "--lr", help="Initial learning rate"),
        weight_decay: float = typer.Option(1e-2, "--wd", help="Weight decay"),
        visualize_lr: bool = typer.Option(False, "--visualize-lr", help="Visualize learning rate schedule"),

        # Device configuration
        gpu_id: str = typer.Option("0", "--gpu", "-g", help="GPU ID(s) to use"),

        # Checkpoint handling
        checkpoint_name: str = typer.Option("none.pth", "--save-as", help="Custom filename to save checkpoint (default: {classifier}_best.pth)"),

        # Noise regularization
        noise_type: NoiseType = typer.Option(NoiseType.none, "--noise-type", help="Type of noise regularization to apply"),
        noise_magnitude: float = typer.Option(0.01, "--noise-magnitude", help="Initial magnitude of noise"),
        noise_schedule: NoiseSchedule = typer.Option(NoiseSchedule.constant, "--noise-schedule", help="Schedule for noise magnitude over time"),
        noise_layers: str = typer.Option(None, "--noise-layers", help="Comma-separated list of layer names to apply noise to (default: all layers)"),
        noise_distribution: NoiseDistribution = typer.Option(NoiseDistribution.gaussian, "--noise-distribution", help="Distribution of noise (gaussian or uniform)"),

):
    # Convert to args-like object for compatibility with existing code
    class Args:
        pass

    args = Args()
    args.data_dir = data_dir
    args.checkpoint_name = checkpoint_name
    args.download_weights = 1 if download_weights else 0
    args.classifier = classifier
    args.model = None
    args.batch_size = batch_size
    args.max_epochs = max_epochs
    args.num_workers = num_workers
    args.gpu_id = gpu_id
    args.learning_rate = learning_rate
    args.weight_decay = weight_decay
    args.noise_type = noise_type
    args.noise_magnitude = noise_magnitude
    args.noise_schedule = noise_schedule
    args.noise_layers = noise_layers
    args.noise_distribution = noise_distribution
    args.subset = subset
    args.test_phase = False
    # Initialize metrics and log handler
    experiment_name = f"{args.classifier}_{args.noise_type}"
    if args.noise_type != NoiseType.none:
        experiment_name += f"_{args.noise_distribution}_{args.noise_magnitude}"

    # Initialize metrics with experiment name for saving
    training_metrics = TrainingMetrics(experiment_name=experiment_name)
    training_metrics.set_config(args)  # Save configuration

    log_handler = LogHandler()


    log_handler.log("SYSTEM", f"Starting CIFAR-10 with {classifier}")

    # Set seeds for reproducibility
    torch.manual_seed(0)
    log_handler.log("SYSTEM", "Random seed set to 0 for reproducibility")

    # Initialize DirectML device
    try:
        device_id = int(args.gpu_id.split(",")[0])  # Use first GPU if multiple specified
        log_handler.log("SYSTEM", f"Initializing DirectML device {device_id}")
        device = torch_directml.device(device_id)
        log_handler.log("SYSTEM", f"Using DirectML device: {device}")
    except Exception as e:
        log_handler.log("ERROR", f"Error initializing DirectML: {e}")
        log_handler.log("SYSTEM", "Falling back to CPU")

        if torch.cuda.is_available():
            device = "cuda"
            print("CUDA USED")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"
            print("MPS USED")
        else:
            device = "cpu"
            print("CPU USED")

    # Download pre-trained weights if requested
    if bool(args.download_weights):
        log_handler.log("SYSTEM", "Downloading pre-trained weights...")
        CIFAR10Data.download_weights()
        return

    # Create model
    log_handler.log("MODEL", f"Creating {args.classifier} model...")
    model = CIFAR10Module(args)

    # Set up data
    log_handler.log("DATA", "Preparing CIFAR-10 dataset...")
    data = CIFAR10Data(args)
    data.prepare_data()
    data.setup()

    # Move model to device
    model.to(device)
    log_handler.log("MODEL", f"Model moved to device: {device}")

    # Check if we're in test mode
    if bool(args.test_phase):
        # Testing mode logic is moved to a separate script test.py
        console.print(Panel("[bold yellow]Use test.py for model evaluation.[/bold yellow]",
                            title="Test Mode", border_style="yellow"))
        return

    # Training mode
    # Create directories for checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    log_handler.log("SYSTEM", "Created checkpoints directory")

    training_metrics.log_weight_norms(model, epoch=0)

    # Get data loaders
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    log_handler.log("DATA", f"Dataset ready with {len(train_loader.dataset)} training samples and {len(val_loader.dataset)} validation samples")

    # Create optimizer and scheduler
    optimizer, scheduler = model.configure_optimizer()
    log_handler.log("OPTIMIZER", f"Created optimizer with lr={args.learning_rate}, weight_decay={args.weight_decay}")

    # Visualize learning rate schedule if requested
    if visualize_lr:
        total_steps = args.max_epochs * len(train_loader)
        # Create a copy of the optimizer and scheduler for visualization
        vis_optimizer, vis_scheduler = model.configure_optimizer()
        log_handler.log("LR_SCHEDULE", "Visualizing learning rate schedule...")

        # Generate visualization
        vis_message = visualize_lr_schedule(vis_optimizer, vis_scheduler, total_steps)
        log_handler.log("LR_SCHEDULE", vis_message)

    # Create initial dashboard layout
    dashboard = create_dashboard(
        training_metrics,
        log_handler,
        args,
        model,
        current_epoch=0,
        max_epochs=args.max_epochs
    )

    # Start training with live display
    with Live(dashboard, refresh_per_second=0.1) as live:
        best_acc = 0.0
        start_time = time.time()

        for epoch in range(args.max_epochs):
            epoch_start_time = time.time()
            log_handler.log("TRAIN", f"Starting epoch {epoch+1}/{args.max_epochs}")

            # Training phase
            model.train()
            # Update noise regularizer epoch counter
            if hasattr(model, 'noise_regularizer') and model.noise_regularizer:
                model.noise_regularizer.update_epoch(epoch)
                current_magnitude = model.noise_regularizer.current_magnitude

                # Log noise metrics
                training_metrics.log_noise_metrics(
                    noise_type=str(model.noise_regularizer.noise_type),
                    noise_magnitude=current_magnitude,
                    epoch=epoch+1
                )
                log_handler.log("NOISE", f"Applying {model.noise_regularizer.noise_type} noise with magnitude {model.noise_regularizer.current_magnitude:.6f}")
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            # Reset batch metrics for new epoch
            training_metrics.reset_batch_metrics()

            for batch_idx, (images, targets) in enumerate(train_loader):
                batch_start = time.time()

                # Move data to device
                images, targets = images.to(device), targets.to(device)

                # Apply weight noise if enabled
                if hasattr(model, 'noise_regularizer') and model.noise_regularizer and model.noise_regularizer.noise_type == NoiseType.weight:

                    permanent = False

                    if not permanent:
                        model.noise_regularizer.save_original_weights(model)

                    # Log weight norms before noise application
                    weight_norms_before = training_metrics.log_weight_norms(model, epoch+1, batch_idx)

                    model.noise_regularizer.apply_weight_noise(model, permanent=permanent)

                    # Log weight norms after noise application
                    weight_norms_after = training_metrics.log_weight_norms(model, epoch+1, batch_idx)

                    # Record noise effects
                    if batch_idx % 50 == 0:  # Don't log too frequently
                        training_metrics.log_noise_metrics(
                            noise_type=str(model.noise_regularizer.noise_type),
                            noise_magnitude=model.noise_regularizer.current_magnitude,
                            norms_before=weight_norms_before,
                            norms_after=weight_norms_after,
                            epoch=epoch+1
                        )

                # Apply label noise if enabled
                if hasattr(model, 'noise_regularizer') and model.noise_regularizer and model.noise_regularizer.noise_type == NoiseType.label:
                    targets = model.noise_regularizer.apply_label_noise(targets)

                # Forward pass
                outputs = model(images)

                # Calculate loss
                loss = model.criterion(outputs, targets)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()

                # Restore original weights if permanent noise is not applied
                if hasattr(model, 'noise_regularizer') and model.noise_regularizer and model.noise_regularizer.noise_type == NoiseType.weight:
                    permanent = False
                    if not permanent:
                        model.noise_regularizer.restore_weights(model)

                # Apply gradient noise if enabled
                if batch_idx % 50 == 0:  # Don't log too frequently
                    grad_norms_before = training_metrics.log_gradient_norms(model, epoch+1, batch_idx)

                # After applying gradient noise:
                if hasattr(model, 'noise_regularizer') and model.noise_regularizer and model.noise_regularizer.noise_type == NoiseType.gradient:
                    model.noise_regularizer.apply_gradient_noise(model)

                    # Log gradient norms after noise application
                    if batch_idx % 50 == 0:  # Don't log too frequently
                        grad_norms_after = training_metrics.log_gradient_norms(model, epoch+1, batch_idx)

                        # Update noise metrics with before/after data
                        training_metrics.log_noise_metrics(
                            noise_type=str(model.noise_regularizer.noise_type),
                            noise_magnitude=model.noise_regularizer.current_magnitude,
                            norms_before=grad_norms_before,
                            norms_after=grad_norms_after,
                            epoch=epoch+1
                        )

                # Update weights
                optimizer.step()

                # Update learning rate scheduler
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]

                # Update statistics
                train_loss += loss.item()
                correct, total = model.calculate_accuracy(outputs, targets)
                train_correct += correct
                train_total += total

                # Track batch metrics
                batch_loss = loss.item()
                batch_acc = correct / total
                training_metrics.add_batch_metrics(batch_loss, batch_acc, current_lr)

                # Update live display every few batches
                if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
                    # Update dashboard
                    dashboard = create_dashboard(
                        training_metrics,
                        log_handler,
                        args,
                        model,
                        current_epoch=epoch+1,
                        max_epochs=args.max_epochs,
                        batch_idx=batch_idx+1,
                        len_train_loader=len(train_loader)
                    )
                    live.update(dashboard)

                # Occasional batch log
                if batch_idx % 50 == 0:
                    batch_time = time.time() - batch_start
                    batch_loss = train_loss / (batch_idx + 1)
                    batch_acc = 100. * train_correct / train_total
                    log_handler.log(
                        "BATCH",
                        f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, "
                        f"Loss: {batch_loss:.4f}, Acc: {batch_acc:.2f}%, "
                        f"LR: {current_lr:.6f}, Time: {batch_time:.2f}s"
                    )

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            log_handler.log("VALID", f"Starting validation for epoch {epoch+1}")

            with torch.no_grad():
                for batch_idx, (images, targets) in enumerate(val_loader):
                    # Move data to device
                    images, targets = images.to(device), targets.to(device)

                    # Forward pass
                    outputs = model(images)
                    loss = model.criterion(outputs, targets)

                    # Update statistics
                    val_loss += loss.item()
                    correct, total = model.calculate_accuracy(outputs, targets)
                    val_correct += correct
                    val_total += total

            # Calculate accuracy and loss metrics
            train_accuracy = 100. * train_correct / train_total
            val_accuracy = 100. * val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            # Add epoch metrics
            training_metrics.add_epoch_metrics(
                epoch=epoch+1,
                train_loss=avg_train_loss,
                train_acc=train_accuracy/100.0,  # Convert to 0-1 range
                val_loss=avg_val_loss,
                val_acc=val_accuracy/100.0,  # Convert to 0-1 range
                lr=current_lr
            )

            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time

            # Log epoch results
            log_handler.log(
                "EPOCH",
                f"Epoch {epoch+1} completed in {epoch_time:.2f}s - "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
            )

            # Save checkpoint if this is the best model
            if val_accuracy > best_acc:
                best_acc = val_accuracy
                if args.checkpoint_name:
                    # Use custom filename if provided
                    checkpoint_filename = args.checkpoint_name
                else:
                    # Create default filename
                    noise_suffix = f"_{args.noise_type}" if args.noise_type != NoiseType.none else ""
                    checkpoint_filename = f"{args.classifier}{noise_suffix}_best.pth"

                checkpoint_path = os.path.join("checkpoints", checkpoint_filename)

                log_handler.log("CHECKPOINT", f"New best accuracy: {val_accuracy:.2f}%, saving to {checkpoint_filename}")
                torch.save(model.state_dict(), checkpoint_path)

            # Update dashboard after validation
            dashboard = create_dashboard(
                training_metrics,
                log_handler,
                args,
                model,
                current_epoch=epoch+1,
                max_epochs=args.max_epochs
            )
            live.update(dashboard)

        # Training complete
        total_time = time.time() - start_time
        log_handler.log(
            "SYSTEM",
            f"Training completed in {datetime.timedelta(seconds=int(total_time))}. "
            f"Best accuracy: {best_acc:.2f}%"
        )

    # Measure model sharpness
    sharpness_data = training_metrics.measure_sharpness(
        model=model,
        criterion=model.criterion,
        val_loader=val_loader,
        device=device
    )

    # Save final metrics
    training_metrics.save_final_metrics(
        best_acc=best_acc / 100.0,  # Convert to 0-1 range
        final_epoch=args.max_epochs,
        total_time=total_time
    )

    # Update the final summary panel to include metrics location
    console.print(Panel(
        f"[bold green]Training completed![/bold green]\n"
        f"[bold]Model:[/bold] {args.classifier}\n"
        f"[bold]Total training time:[/bold] {datetime.timedelta(seconds=int(total_time))}\n"
        f"[bold]Best validation accuracy:[/bold] {best_acc:.2f}%\n"
        f"[bold]Checkpoint saved to:[/bold] checkpoints/{args.classifier}_best.pth\n"
        f"[bold]Metrics saved to:[/bold] {training_metrics.metrics_dir}",
        title="Training Results",
        border_style="green"
    ))

if __name__ == "__main__":
    typer.run(main)