import os
import time
import datetime
import typer
from enum import Enum
from typing import Optional, List, Dict
import numpy as np

import torch
import torch_directml

from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich import box
from rich.columns import Columns
from rich.align import Align
from rich.traceback import install as install_rich_traceback
import matplotlib.pyplot as plt

from data import CIFAR10Data
from module import CIFAR10Module
from noise_regularization import *

# Install rich traceback handler
install_rich_traceback(show_locals=True)

# Initialize console
console = Console()

# Define color theme
THEME = {
    "title": "magenta",
    "heading": "yellow",
    "metrics": "cyan",
    "good": "green",
    "warning": "yellow",
    "bad": "red",
    "accent": "blue",
    "graph_dots": "bright_cyan",
    "graph_line": "blue"
}

class Classifier(str, Enum):
    vgg11_bn = "vgg11_bn"
    vgg13_bn = "vgg13_bn"
    vgg16_bn = "vgg16_bn"
    vgg19_bn = "vgg19_bn"
    resnet18 = "resnet18"
    resnet34 = "resnet34"
    resnet50 = "resnet50"
    densenet121 = "densenet121"
    densenet161 = "densenet161"
    densenet169 = "densenet169"
    mobilenet_v2 = "mobilenet_v2"
    googlenet = "googlenet"
    inception_v3 = "inception_v3"

class LoggerType(str, Enum):
    tensorboard = "tensorboard"
    wandb = "wandb"

class TrainingMetrics:
    """Class to track and store training metrics over time"""
    def __init__(self, max_history=100):
        self.max_history = max_history
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.lr_history = []
        self.epochs = []

        # For batch-level tracking
        self.current_epoch_losses = []
        self.current_epoch_accs = []
        self.current_batch_lrs = []

    def add_batch_metrics(self, loss, acc, lr):
        """Add batch-level metrics"""
        self.current_epoch_losses.append(loss)
        self.current_epoch_accs.append(acc)
        self.current_batch_lrs.append(lr)

    def add_epoch_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        """Add epoch-level metrics"""
        self.epochs.append(epoch)
        self.train_loss_history.append(train_loss)
        self.train_acc_history.append(train_acc)
        self.val_loss_history.append(val_loss)
        self.val_acc_history.append(val_acc)
        self.lr_history.append(lr)

        # Trim if exceeding max history
        if len(self.epochs) > self.max_history:
            self.epochs = self.epochs[-self.max_history:]
            self.train_loss_history = self.train_loss_history[-self.max_history:]
            self.train_acc_history = self.train_acc_history[-self.max_history:]
            self.val_loss_history = self.val_loss_history[-self.max_history:]
            self.val_acc_history = self.val_acc_history[-self.max_history:]
            self.lr_history = self.lr_history[-self.max_history:]

    def reset_batch_metrics(self):
        """Reset batch-level metrics for a new epoch"""
        self.current_epoch_losses = []
        self.current_epoch_accs = []
        self.current_batch_lrs = []

    def get_current_epoch_avg_metrics(self):
        """Get average metrics for the current epoch"""
        if not self.current_epoch_losses:
            return 0.0, 0.0, 0.0

        avg_loss = sum(self.current_epoch_losses) / len(self.current_epoch_losses)
        avg_acc = sum(self.current_epoch_accs) / len(self.current_epoch_accs)
        last_lr = self.current_batch_lrs[-1] if self.current_batch_lrs else 0.0

        return avg_loss, avg_acc, last_lr

class LogHandler:
    """Simple logging system with timestamps"""
    def __init__(self, max_entries=100):
        self.max_entries = max_entries
        self.logs = []

    def log(self, source, message, level="INFO"):
        """Add a log entry with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        entry = {
            "time": timestamp,
            "source": source.upper(),
            "level": level.upper(),
            "message": message
        }
        self.logs.append(entry)

        # Trim if exceeding max entries
        if len(self.logs) > self.max_entries:
            self.logs = self.logs[-self.max_entries:]

    def get_logs_table(self):
        """Return a Rich table of log entries"""
        log_table = Table(show_header=True, box=box.SIMPLE, expand=True)
        log_table.add_column("Time", style="dim")
        log_table.add_column("Source", style="yellow")
        log_table.add_column("Message", style="white", ratio=3)

        for entry in self.logs:
            source_style = "yellow"
            if entry["level"] == "ERROR":
                source_style = "red"
            elif entry["level"] == "WARNING":
                source_style = "orange3"

            log_table.add_row(
                entry["time"],
                f"[{source_style}]{entry['source']}[/{source_style}]",
                entry["message"]
            )

        return log_table

def create_accuracy_graph(metrics, width=60, height=10):
    """Create a simple ASCII graph of accuracy history"""
    if not metrics.train_acc_history:
        return Text("No data yet", style="dim")

    # Setup the canvas
    canvas = [[" " for _ in range(width)] for _ in range(height)]

    # Calculate scaling factors
    y_max = max(max(metrics.train_acc_history, default=0), max(metrics.val_acc_history, default=0)) * 1.1
    y_max = max(y_max, 0.1)  # Ensure minimum scale
    y_min = 0

    x_values = list(range(len(metrics.train_acc_history)))
    if not x_values:
        return Text("No data yet", style="dim")

    x_max = max(x_values)
    x_min = 0

    # Scale factors
    if x_max > x_min:
        x_scale = (width - 1) / (x_max - x_min)
    else:
        x_scale = 1

    y_scale = (height - 1) / (y_max - y_min)

    # Plot training accuracy
    for i in range(len(metrics.train_acc_history) - 1):
        x1 = int((i - x_min) * x_scale)
        y1 = height - 1 - int((metrics.train_acc_history[i] - y_min) * y_scale)
        x2 = int((i + 1 - x_min) * x_scale)
        y2 = height - 1 - int((metrics.train_acc_history[i + 1] - y_min) * y_scale)

        # Ensure within bounds
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        # Plot point
        canvas[y1][x1] = "*"

        # Connect with line if close
        if abs(x2 - x1) <= 1:
            canvas[y2][x2] = "*"

    # Plot validation accuracy (if exists) with different marker
    if metrics.val_acc_history:
        for i in range(len(metrics.val_acc_history)):
            x = int((i - x_min) * x_scale)
            y = height - 1 - int((metrics.val_acc_history[i] - y_min) * y_scale)

            # Ensure within bounds
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))

            # Mark with different symbol
            canvas[y][x] = "+"

    # Convert canvas to text
    result = ""
    for row in canvas:
        result += "".join(row) + "\n"

    # Add y-axis labels
    y_axis = Text.assemble(
        f"{y_max*100:.1f}%\n",
        "\n" * (height // 2 - 1),
        f"{((y_max+y_min)/2)*100:.1f}%\n",
        "\n" * (height // 2 - 1),
        f"{y_min*100:.1f}%",
        )

    # Create a separator
    separator = Text("|\n" * height)

    # Create x-axis
    x_axis = "-" * width

    # Get the last training and validation accuracy values
    train_acc_last = 0
    if metrics.train_acc_history:
        train_acc_last = metrics.train_acc_history[-1] * 100

    val_acc_last = 0
    if metrics.val_acc_history:
        val_acc_last = metrics.val_acc_history[-1] * 100

    # Combine into a single layout
    graph = Text.assemble(
        Text(result, style=THEME["graph_dots"]),
        Text(f"\n{x_axis}", style=THEME["accent"]),
        Text.from_markup(f"\nEpoch: [b]{len(metrics.epochs)}[/b]"),
    )

    # Add training accuracy markup safely
    graph.append_text(Text.from_markup(f"Train: [green]{train_acc_last:.2f}%[/green] "))

    # Add validation accuracy markup if available
    if metrics.val_acc_history:
        graph.append_text(Text.from_markup(f"Val: [blue]{val_acc_last:.2f}%[/blue]"))

    return graph
def create_loss_graph(metrics, width=60, height=10):
    """Create a simple ASCII graph of loss history"""
    if not metrics.train_loss_history:
        return Text("No data yet", style="dim")

    # Setup the canvas
    canvas = [[" " for _ in range(width)] for _ in range(height)]

    # Calculate scaling factors
    y_max = max(max(metrics.train_loss_history, default=0), max(metrics.val_loss_history, default=0)) * 1.1
    y_max = max(y_max, 0.1)  # Ensure minimum scale
    y_min = 0

    x_values = list(range(len(metrics.train_loss_history)))
    if not x_values:
        return Text("No data yet", style="dim")

    x_max = max(x_values)
    x_min = 0

    # Scale factors
    if x_max > x_min:
        x_scale = (width - 1) / (x_max - x_min)
    else:
        x_scale = 1

    y_scale = (height - 1) / (y_max - y_min)

    # Plot training loss
    for i in range(len(metrics.train_loss_history) - 1):
        x1 = int((i - x_min) * x_scale)
        y1 = height - 1 - int((metrics.train_loss_history[i] - y_min) * y_scale)
        x2 = int((i + 1 - x_min) * x_scale)
        y2 = height - 1 - int((metrics.train_loss_history[i + 1] - y_min) * y_scale)

        # Ensure within bounds
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        # Plot point
        canvas[y1][x1] = "*"

        # Connect with line if close
        if abs(x2 - x1) <= 1:
            canvas[y2][x2] = "*"

    # Plot validation loss (if exists) with different marker
    if metrics.val_loss_history:
        for i in range(len(metrics.val_loss_history)):
            x = int((i - x_min) * x_scale)
            y = height - 1 - int((metrics.val_loss_history[i] - y_min) * y_scale)

            # Ensure within bounds
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))

            # Mark with different symbol
            canvas[y][x] = "+"

    # Convert canvas to text
    result = ""
    for row in canvas:
        result += "".join(row) + "\n"

    # Add x-axis
    x_axis = "-" * width

    # Get the last training and validation loss values
    train_loss_last = 0
    if metrics.train_loss_history:
        train_loss_last = metrics.train_loss_history[-1]

    val_loss_last = 0
    if metrics.val_loss_history:
        val_loss_last = metrics.val_loss_history[-1]

    # Combine into a single layout
    graph = Text.assemble(
        Text(result, style=THEME["graph_dots"]),
        Text(f"\n{x_axis}", style=THEME["accent"]),
        Text.from_markup(f"\nEpoch: [b]{len(metrics.epochs)}[/b]"),
    )

    # Add training loss markup safely
    graph.append_text(Text.from_markup(f"Train: [red]{train_loss_last:.4f}[/red] "))

    # Add validation loss markup if available
    if metrics.val_loss_history:
        graph.append_text(Text.from_markup(f"Val: [blue]{val_loss_last:.4f}[/blue]"))

    return graph

def create_lr_graph(metrics, width=60, height=6):
    """Create a simple ASCII graph of learning rate history"""
    if not metrics.lr_history:
        return Text("No data yet", style="dim")

    # Setup the canvas
    canvas = [[" " for _ in range(width)] for _ in range(height)]

    # Calculate scaling factors
    y_max = max(metrics.lr_history, default=0) * 1.1
    y_max = max(y_max, 1e-6)  # Ensure minimum scale
    y_min = min(metrics.lr_history, default=0)

    x_values = list(range(len(metrics.lr_history)))
    if not x_values:
        return Text("No data yet", style="dim")

    x_max = max(x_values)
    x_min = 0

    # Scale factors
    if x_max > x_min:
        x_scale = (width - 1) / (x_max - x_min)
    else:
        x_scale = 1

    if y_max > y_min:
        y_scale = (height - 1) / (y_max - y_min)
    else:
        y_scale = 1

    # Plot learning rate
    for i in range(len(metrics.lr_history) - 1):
        x1 = int((i - x_min) * x_scale)
        y1 = height - 1 - int((metrics.lr_history[i] - y_min) * y_scale)
        x2 = int((i + 1 - x_min) * x_scale)
        y2 = height - 1 - int((metrics.lr_history[i + 1] - y_min) * y_scale)

        # Ensure within bounds
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        # Plot point
        canvas[y1][x1] = "·"

        # Connect with line if close
        if abs(x2 - x1) <= 1:
            canvas[y2][x2] = "·"

    # Convert canvas to text
    result = ""
    for row in canvas:
        result += "".join(row) + "\n"

    # Add x-axis
    x_axis = "-" * width

    # Get last learning rate
    lr_last = 0
    if metrics.lr_history:
        lr_last = metrics.lr_history[-1]

    # Combine into a single layout
    graph = Text.assemble(
        Text(result, style=THEME["accent"]),
        Text(f"\n{x_axis}", style=THEME["accent"]),
    )

    # Add LR value safely
    graph.append_text(Text.from_markup(f"\nLR: [yellow]{lr_last:.6f}[/yellow]"))

    return graph

def create_dashboard(metrics, log_handler, args, model, current_epoch, max_epochs, batch_idx=None, len_train_loader=None):
    """Create a dashboard layout with all the information"""
    # Create main layout
    layout = Layout()

    # Check if we have a noise regularizer
    has_noise_regularizer = hasattr(model, 'noise_regularizer') and model.noise_regularizer is not None

    # Split into header, body and footer
    layout.split(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=4)
    )

    # Header - Title and info
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    header_layout = Layout()
    header_layout.split_row(
        Layout(Panel(f"[bold {THEME['title']}]CIFAR-10 Training with {args.classifier}[/]", border_style=THEME["accent"]), ratio=2),
        Layout(Align.right(Text.assemble(
            Text.from_markup(f"[{THEME['heading']}]Date:[/{THEME['heading']}] {current_time}"),
        )), ratio=1)
    )
    layout["header"].update(header_layout)

    # New layout organization: Graphs in top row, text-based info in bottom row
    layout["body"].split(
        Layout(name="graphs_row", ratio=1),  # Top row for all graphs
        Layout(name="info_row", ratio=1)     # Bottom row for all text-based info
    )

    # Split the graphs row
    if has_noise_regularizer:
        # When we have noise regularizer, make room for its visualization
        layout["body"]["graphs_row"].split_row(
            Layout(name="accuracy_graph", ratio=1),
            Layout(name="loss_graph", ratio=1),
            Layout(name="noise_magnitude_graph", ratio=1),
            Layout(name="lr_graph", ratio=1)
        )
    else:
        # Without noise regularizer, just the standard graphs
        layout["body"]["graphs_row"].split_row(
            Layout(name="accuracy_graph", ratio=1),
            Layout(name="loss_graph", ratio=1),
            Layout(name="lr_graph", ratio=1)
        )

    # Split the info row
    if has_noise_regularizer:
        layout["body"]["info_row"].split_row(
            Layout(name="training_metrics", ratio=1),
            Layout(name="model_info", ratio=1),
            Layout(name="noise_metrics", ratio=1),
            Layout(name="log", ratio=2)
        )
    else:
        layout["body"]["info_row"].split_row(
            Layout(name="training_metrics", ratio=1),
            Layout(name="model_info", ratio=1),
            Layout(name="log", ratio=2)
        )

    # Current training metrics
    avg_loss, avg_acc, last_lr = metrics.get_current_epoch_avg_metrics()

    # Format epoch/batch info
    if batch_idx is not None and len_train_loader is not None:
        epoch_info = f"Epoch: {current_epoch}/{max_epochs} [{batch_idx}/{len_train_loader}]"
    else:
        epoch_info = f"Epoch: {current_epoch}/{max_epochs}"

    # Display current metrics
    metrics_table = Table(show_header=True, box=box.SIMPLE, title=epoch_info)
    metrics_table.add_column("Metric", style=THEME["heading"])
    metrics_table.add_column("Value", style=THEME["metrics"])

    metrics_table.add_row("Current Loss", f"{avg_loss:.4f}")
    metrics_table.add_row("Current Accuracy", f"{avg_acc*100:.2f}%")
    metrics_table.add_row("Learning Rate", f"{last_lr:.6f}")

    if metrics.val_acc_history:
        best_val_acc = max(metrics.val_acc_history)
        best_val_epoch = metrics.epochs[metrics.val_acc_history.index(best_val_acc)]
        metrics_table.add_row("Best Validation Acc", f"[{THEME['good']}]{best_val_acc*100:.2f}%[/{THEME['good']}] (Epoch {best_val_epoch})")

    # Model info
    model_info = Table(show_header=True, box=box.SIMPLE, title="Model Information")
    model_info.add_column("Property", style=THEME["heading"])
    model_info.add_column("Value", style=THEME["metrics"])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_info.add_row("Architecture", str(args.classifier))
    model_info.add_row("Total Parameters", f"{total_params:,}")
    model_info.add_row("Trainable Parameters", f"{trainable_params:,}")
    model_info.add_row("Batch Size", str(args.batch_size))
    model_info.add_row("Workers", str(args.num_workers))
    model_info.add_row("Initial LR", str(args.learning_rate))
    model_info.add_row("Weight Decay", str(args.weight_decay))

    # Create accuracy and loss graphs
    accuracy_graph = Panel(
        create_accuracy_graph(metrics, width=80, height=15),  # Increased height
        title="Accuracy History",
        border_style=THEME["good"],
        expand=True
    )

    loss_graph = Panel(
        create_loss_graph(metrics, width=80, height=15),  # Increased height
        title="Loss History",
        border_style=THEME["bad"],
        expand=True
    )

    # Create LR graph with more height
    lr_graph = Panel(
        create_lr_graph(metrics, width=80, height=15),  # Increased height
        title="Learning Rate Schedule",
        border_style=THEME["accent"],
        expand=True
    )

    # Create log panel
    log_panel = Panel(
        log_handler.get_logs_table(),
        title="Training Log",
        border_style="green",
        expand=True
    )

    # Progress bars for footer
    progress_table = Table(show_header=False, box=None, expand=True)
    progress_table.add_column("Label", style=THEME["heading"])
    progress_table.add_column("Progress", ratio=3)
    progress_table.add_column("Info", justify="right")

    # Epoch progress
    epoch_progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style=THEME["good"]),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        expand=True
    )
    epoch_task = epoch_progress.add_task("Epoch Progress", total=max_epochs, completed=current_epoch)

    # Batch progress (if applicable)
    if batch_idx is not None and len_train_loader is not None:
        batch_progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style=THEME["accent"]),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            expand=True
        )
        batch_task = batch_progress.add_task("Batch Progress", total=len_train_loader, completed=batch_idx)
        progress_table.add_row("", batch_progress, "")

    progress_table.add_row("", epoch_progress, "")

    # Update all layout sections - graphs row
    layout["body"]["graphs_row"]["accuracy_graph"].update(accuracy_graph)
    layout["body"]["graphs_row"]["loss_graph"].update(loss_graph)

    # Information row
    layout["body"]["info_row"]["training_metrics"].update(
        Panel(metrics_table, title="Training Metrics", border_style=THEME["accent"]))
    layout["body"]["info_row"]["model_info"].update(
        Panel(model_info, title="Model Configuration", border_style=THEME["heading"]))
    layout["body"]["info_row"]["log"].update(log_panel)

    # Add noise regularization elements if available
    if has_noise_regularizer:
        # Create default table and graph for noise metrics
        noise_metrics_table = Text("Collecting noise metrics...", style="dim")
        noise_graph = Text("Collecting noise data...", style="dim")

        # Try to get actual metrics if available
        try:
            noise_metrics_table = model.noise_regularizer.get_metrics_table()
            noise_graph = model.noise_regularizer.create_noise_magnitude_graph(width=80, height=15)  # Increased height
        except Exception as e:
            # If there's an error, use the default text
            pass

        # Create panels with the content
        noise_metrics_panel = Panel(
            noise_metrics_table,
            title=f"{model.noise_regularizer.noise_type.capitalize()} Noise Metrics",
            border_style="yellow",
            expand=True
        )

        noise_graph_panel = Panel(
            noise_graph,
            title="Noise Magnitude History",
            border_style="cyan",
            expand=True
        )

        # Update noise-specific sections
        layout["body"]["graphs_row"]["noise_magnitude_graph"].update(noise_graph_panel)
        layout["body"]["info_row"]["noise_metrics"].update(noise_metrics_panel)

    # Put LR graph in the graph row rather than footer
    layout["body"]["graphs_row"]["lr_graph"].update(lr_graph)

    # Update footer with just progress
    layout["footer"].update(progress_table)

    return layout

def visualize_lr_schedule(optimizer, scheduler, steps, save_path="lr_schedule.png"):
    """Plot the learning rate schedule"""
    lrs = []
    for i in range(steps):
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])

    plt.figure(figsize=(10, 5))
    plt.plot(range(steps), lrs)
    plt.xlabel('Steps')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    return f"Learning rate schedule saved to {save_path}"

def main(
        data_dir: str = "./data/cifar10",
        download_weights: bool = typer.Option(False, "--download-weights", "-w", help="Download pre-trained weights"),
        test_phase: bool = typer.Option(False, "--test", "-t", help="Run in test phase"),
        dev: bool = typer.Option(False, "--dev", "-d", help="Run in development mode"),
        logger_type: LoggerType = typer.Option(LoggerType.tensorboard, "--logger", "-l", help="Logger to use"),
        classifier: Classifier = typer.Option(Classifier.resnet18, "--classifier", "-c", help="Classifier model to use"),
        pretrained: bool = typer.Option(False, "--pretrained", "-p", help="Use pre-trained model"),
        batch_size: int = typer.Option(128, "--batch-size", "-b", help="Batch size for training"),
        max_epochs: int = typer.Option(100, "--epochs", "-e", help="Maximum number of epochs"),
        num_workers: int = typer.Option(4, "--workers", "-w", help="Number of data loading workers"),
        gpu_id: str = typer.Option("0", "--gpu", "-g", help="GPU ID(s) to use"),
        learning_rate: float = typer.Option(1e-2, "--lr", help="Initial learning rate"),
        weight_decay: float = typer.Option(1e-2, "--wd", help="Weight decay"),
        visualize_lr: bool = typer.Option(False, "--visualize-lr", help="Visualize learning rate schedule"),
        noise_type: NoiseType = typer.Option(NoiseType.none, "--noise-type", help="Type of noise regularization to apply"),
        noise_magnitude: float = typer.Option(0.01, "--noise-magnitude",help="Initial magnitude of noise"),
        noise_schedule: NoiseSchedule = typer.Option(NoiseSchedule.constant, "--noise-schedule",help="Schedule for noise magnitude over time"),
        noise_layers: str = typer.Option(None, "--noise-layers",help="Comma-separated list of layer names to apply noise to (default: all layers)"),
        ):
    # Convert to args-like object for compatibility with existing code
    class Args:
        pass

    args = Args()
    args.data_dir = data_dir
    args.download_weights = 1 if download_weights else 0
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
    args.noise_type = noise_type
    args.noise_magnitude = noise_magnitude
    args.noise_schedule = noise_schedule
    args.noise_layers = noise_layers


    # Initialize metrics and log handler
    training_metrics = TrainingMetrics()
    log_handler = LogHandler()

    log_handler.log("SYSTEM", f"Starting CIFAR-10 training with {classifier}")
    log_handler.log("CONFIG", f"Batch size: {batch_size}, Learning rate: {learning_rate}, Weight decay: {weight_decay}")

    if bool(args.download_weights):
        log_handler.log("SYSTEM", "Downloading pre-trained weights...")
        CIFAR10Data.download_weights()
    else:
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
            device = torch.device("cpu")

        # Create model and data modules
        log_handler.log("MODEL", f"Creating {args.classifier} model...")
        model = CIFAR10Module(args)
        data = CIFAR10Data(args)

        # Move model to device
        model.to(device)
        log_handler.log("MODEL", f"Model moved to device: {device}")

        # Setup data loaders
        log_handler.log("DATA", "Preparing CIFAR-10 dataset...")
        data.prepare_data()
        data.setup()
        train_loader = data.train_dataloader()
        val_loader = data.val_dataloader()
        log_handler.log("DATA", f"Dataset ready with {len(train_loader.dataset)} training samples and {len(val_loader.dataset)} validation samples")

        # Create optimizer and scheduler
        optimizer, scheduler = model.configure_optimizer()
        log_handler.log("OPTIMIZER", f"Created optimizer with lr={args.learning_rate}, weight_decay={args.weight_decay}")

        # Create directories for checkpoints
        os.makedirs("checkpoints", exist_ok=True)
        log_handler.log("SYSTEM", "Created checkpoints directory")

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
                    if hasattr(model, 'noise_regularizer') and model.noise_regularizer and model.noise_regularizer.noise_type == NoiseType.weight:
                        model.noise_regularizer.apply_weight_noise(model, permanent=False)
                    # Forward pass
                    outputs = model(images)


                    if hasattr(model, 'noise_regularizer') and model.noise_regularizer and model.noise_regularizer.noise_type == NoiseType.label:
                        targets = model.noise_regularizer.apply_label_noise(targets)
                    loss = model.criterion(outputs, targets)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()

                    if hasattr(model, 'noise_regularizer') and model.noise_regularizer and model.noise_regularizer.noise_type == NoiseType.gradient:
                        model.noise_regularizer.apply_gradient_noise(model)
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
                    log_handler.log("CHECKPOINT", f"New best accuracy: {val_accuracy:.2f}%")
                    torch.save(model.state_dict(), f"checkpoints/{args.classifier}_best.pth")

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