from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich import box
from rich.align import Align
import matplotlib.pyplot as plt
import time

from .configs.config import THEME

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
