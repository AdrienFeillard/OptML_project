import os
import typer
import torch
import torch_directml
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

# Make sure Python can find modules in the current directory
import sys
sys.path.append(os.getcwd())

# Import from modules
from core.data import CIFAR10Data
from core.module import CIFAR10Module
from utils.configs.config import Classifier, NoiseDistribution
from core.noise_regularization import NoiseType, NoiseSchedule

# Initialize console
console = Console()

def main(
        checkpoint_path: str = typer.Option("checkpoints/resnet18_best.pth", "--checkpoint", "-c", help="Path to model checkpoint"),
        classifier: str = typer.Option("resnet18", "--classifier", "-m", help="Classifier model to test"),
        data_dir: str = typer.Option("./data/cifar10", "--data-dir", "-d", help="Path to data directory"),
        batch_size: int = typer.Option(128, "--batch-size", "-b", help="Batch size for testing"),
        num_workers: int = typer.Option(4, "--workers", "-w", help="Number of data loading workers"),
        gpu_id: str = typer.Option("0", "--gpu", "-g", help="GPU ID to use"),
        noise_distribution: NoiseDistribution = typer.Option(NoiseDistribution.gaussian, "--noise-distribution", help="Distribution of noise (gaussian or uniform)"),

):

    console.print(Panel(f"[bold cyan]Testing {classifier} model[/bold cyan]",
                        title="Test Mode", border_style="cyan"))

    # Set up args for model
    class Args:
        pass

    args = Args()
    args.classifier = classifier
    args.data_dir = data_dir
    args.batch_size = batch_size
    args.num_workers = num_workers
    args.gpu_id = gpu_id
    args.learning_rate = 0.01  # Default value, not used during testing
    args.weight_decay = 0.01   # Default value, not used during testing
    args.max_epochs = 100      # Default value, not used during testing
    args.noise_type = NoiseType.none  # No noise during testing
    args.noise_magnitude = 0.0
    args.noise_schedule = NoiseSchedule.constant
    args.noise_layers = None
    args.noise_distribution = noise_distribution

    # Set up device
    try:
        device_id = int(args.gpu_id.split(",")[0])
        console.print(f"Initializing DirectML device {device_id}")
        device = torch_directml.device(device_id)
        console.print(f"Using DirectML device: {device}")
    except Exception as e:
        console.print(f"[bold red]Error initializing DirectML: {e}[/bold red]")
        console.print("Falling back to CPU")
        device = torch.device("cpu")

    # Create model
    console.print(f"Creating {args.classifier} model...")
    model = CIFAR10Module(args)
    model.to(device)

    # Load checkpoint
    console.print(f"Loading checkpoint from: {checkpoint_path}")
    try:
        model.load_state_dict(torch.load(checkpoint_path))
        console.print("[bold green]Checkpoint loaded successfully![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error loading checkpoint: {e}[/bold red]")
        console.print(f"[yellow]Make sure the path is correct and the file exists.[/yellow]")
        return

    # Set model to evaluation mode
    model.eval()

    # Set up data
    data = CIFAR10Data(args)
    data.prepare_data()
    data.setup()
    test_loader = data.test_dataloader()
    console.print(f"Test dataset contains {len(test_loader.dataset)} samples")

    # Prepare for evaluation
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    # Initialize per-class metrics
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    class_correct = [0] * 10
    class_total = [0] * 10

    # Evaluate the model
    console.print("\n[bold cyan]Starting evaluation...[/bold cyan]")
    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeRemainingColumn(),
    ) as progress:
        test_task = progress.add_task("[cyan]Testing...", total=len(test_loader))

        # Evaluation loop
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(test_loader):
                # Move data to device
                images, targets = images.to(device), targets.to(device)

                # Forward pass
                outputs = model(images)
                loss = model.criterion(outputs, targets)

                # Update statistics
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

                # Per-class accuracy
                c = (predicted == targets).squeeze()
                for i in range(targets.size(0)):
                    label = targets[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

                # Update progress
                progress.update(test_task, advance=1)

    # Calculate final metrics
    avg_test_loss = test_loss / len(test_loader)
    accuracy = 100.0 * test_correct / test_total

    # Display results
    results_table = Table(title="Test Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")

    results_table.add_row("Average Loss", f"{avg_test_loss:.4f}")
    results_table.add_row("Accuracy", f"{accuracy:.2f}%")
    results_table.add_row("Correct/Total", f"{test_correct}/{test_total}")

    console.print(results_table)

    # Display per-class accuracy
    class_table = Table(title="Per-Class Accuracy")
    class_table.add_column("Class", style="cyan")
    class_table.add_column("Accuracy", style="green")
    class_table.add_column("Correct/Total", style="dim")

    for i in range(10):
        class_acc = 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        class_table.add_row(
            classes[i],
            f"{class_acc:.2f}%",
            f"{int(class_correct[i])}/{class_total[i]}"
        )

    console.print(class_table)

    # Final summary
    console.print(Panel(
        f"[bold green]Testing completed![/bold green]\n"
        f"[bold]Model:[/bold] {args.classifier}\n"
        f"[bold]Test accuracy:[/bold] {accuracy:.2f}%\n"
        f"[bold]Test loss:[/bold] {avg_test_loss:.4f}",
        title="Test Results",
        border_style="green"
    ))

if __name__ == "__main__":
    typer.run(main)