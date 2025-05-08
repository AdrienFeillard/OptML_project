import os
import time
import datetime
import typer
from typing import List
import torch
#import torch_directml
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.traceback import install as install_rich_traceback

# Import from reorganized modules
from utils.configs.config import Classifier, LoggerType
from core.data import CIFAR10Data
from core.module import CIFAR10Module
from core.perturbedModule import PerturbedCIFAR10Module
from core.noise_regularization import NoiseType, NoiseSchedule
from utils.logging import LogHandler
from utils.metrics import TrainingMetrics
from utils.visualization import create_dashboard, visualize_lr_schedule

# Install rich traceback handler
install_rich_traceback(show_locals=True)

# Initialize console
console = Console()
def main(
        data_dir: str = "./data/cifar10",
        data_subset: float = typer.Option(1., "--subset", "-s", help="Subset of data to use (1: 10%, 2: 20%, 3: 50%, 4: 100%)"),
        download_weights: bool = typer.Option(False, "--download-weights", "-w", help="Download pre-trained weights"),
        test_phase: bool = typer.Option(False, "--test", "-t", help="Run in test phase"),
        checkpoint_path: str = typer.Option("checkpoints/resnet18_best.pth", "--checkpoint", help="Path to model checkpoint for testing"),
        checkpoint_name: str = typer.Option(None, "--save-as", help="Custom filename to save checkpoint (default: {classifier}_best.pth)"),
        dev: bool = typer.Option(False, "--dev", "-d", help="Run in development mode"),
        logger_type: LoggerType = typer.Option(LoggerType.tensorboard, "--logger", "-l", help="Logger to use"),
        classifier: str = typer.Option("resnet18", "--classifier", "-c", help="Classifier model to use"),
        pretrained: bool = typer.Option(False, "--pretrained", "-p", help="Use pre-trained model"),
        batch_size: int = typer.Option(128, "--batch-size", "-b", help="Batch size for training"),
        max_epochs: int = typer.Option(100, "--epochs", "-e", help="Maximum number of epochs"),
        num_workers: int = typer.Option(4, "--workers", "-w", help="Number of data loading workers"),
        gpu_id: str = typer.Option("0", "--gpu", "-g", help="GPU ID(s) to use"),
        learning_rate: float = typer.Option(1e-2, "--lr", help="Initial learning rate"),
        weight_decay: float = typer.Option(1e-2, "--wd", help="Weight decay"),
        visualize_lr: bool = typer.Option(False, "--visualize-lr", help="Visualize learning rate schedule"),
        noise_position: str = typer.Option(None, "--noise-position", help="Position to apply noise regularization"),
        noise_type: str = typer.Option(None, "--noise-type", help="Type of noise regularization to apply : None, gaussian or uniform"),
        noise_std: float = typer.Option(0.01, "--noise-std", help="Initial standard deviacion of noise"),
        noise_schedule: str = typer.Option(None, "--noise-schedule",help="Schedule for noise magnitude over time"),
        noise_layer: List[str] = typer.Option(None, "--noise-layer",help="List of layer names to apply noise to (default: all layers). Example of use : --noise-layers conv1 --noise-layers conv2'"),
        ):
    # Convert to args-like object for compatibility with existing code
    class Args:
        pass

    args = Args()
    args.data_dir = data_dir
    args.checkpoint_name = checkpoint_name
    args.download_weights = 1 if download_weights else 0
    args.subset = data_subset
    args.test_phase = 1 if test_phase else 0
    args.dev = 1 if dev else 0
    args.logger = logger_type
    args.classifier = classifier
    args.model = None
    args.pretrained = 1 if pretrained else 0
    args.batch_size = batch_size
    args.max_epochs = max_epochs
    args.num_workers = num_workers
    args.gpu_id = gpu_id
    args.learning_rate = learning_rate
    args.weight_decay = weight_decay
    args.noise_position = noise_position
    args.noise_type = noise_type
    args.noise_std = noise_std
    args.noise_schedule = noise_schedule
    args.noise_layer = noise_layer

    args.checkpoint_path = checkpoint_path

    # Initialize metrics and log handler
    training_metrics = TrainingMetrics()
    log_handler = LogHandler()

    log_handler.log("SYSTEM", f"Starting CIFAR-10 with {classifier}")

    # Set seeds for reproducibility
    torch.manual_seed(0)
    log_handler.log("SYSTEM", "Random seed set to 0 for reproducibility")

    # Initialize DirectML device
    # try:
    #     device_id = int(args.gpu_id.split(",")[0])  # Use first GPU if multiple specified
    #     log_handler.log("SYSTEM", f"Initializing DirectML device {device_id}")
    #     device = torch_directml.device(device_id)
    #     log_handler.log("SYSTEM", f"Using DirectML device: {device}")
    # except Exception as e:
        #log_handler.log("ERROR", f"Error initializing DirectML: {e}")
    log_handler.log("SYSTEM", "Falling back to CPU")
    device = torch.device("cpu")

    # Download pre-trained weights if requested
    if bool(args.download_weights):
        log_handler.log("SYSTEM", "Downloading pre-trained weights...")
        CIFAR10Data.download_weights()
        return

    # Create model
    log_handler.log("MODEL", f"Creating {args.classifier} model...")
    #model = CIFAR10Module(args)
    model = PerturbedCIFAR10Module(args)

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
                #if hasattr(model, 'noise_regularizer') and model.noise_regularizer and model.noise_regularizer.noise_type == NoiseType.weight:
                #    model.noise_regularizer.apply_weight_noise(model, permanent=False)

                # Forward pass
                outputs = model(images)

                # Apply label noise if enabled
                #if hasattr(model, 'noise_regularizer') and model.noise_regularizer and model.noise_regularizer.noise_type == NoiseType.label:
                #    targets = model.noise_regularizer.apply_label_noise(targets)

                # Calculate loss
                loss = model.criterion(outputs, targets)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()

                # Apply gradient noise if enabled
                #if hasattr(model, 'noise_regularizer') and model.noise_regularizer and model.noise_regularizer.noise_type == NoiseType.gradient:
                #    model.noise_regularizer.apply_gradient_noise(model)

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

    # Final summary panel
    console.print(Panel(
        f"[bold green]Training completed![/bold green]\n"
        f"[bold]Model:[/bold] {args.classifier}\n"
        f"[bold]Total training time:[/bold] {datetime.timedelta(seconds=int(total_time))}\n"
        f"[bold]Best validation accuracy:[/bold] {best_acc:.2f}%\n"
        f"[bold]Checkpoint saved to:[/bold] checkpoints/{args.classifier}_best.pth",
        title="Training Results",
        border_style="green"
    ))

if __name__ == "__main__":
    typer.run(main)