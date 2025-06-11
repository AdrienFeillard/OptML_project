from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich import box
from rich.align import Align
import matplotlib.pyplot as plt
import time
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich import box
from rich.align import Align
from rich.console import Group # Added for grouping progress bars
import matplotlib.pyplot as plt # Keep for visualize_lr_schedule
import time
from typing import Dict, Any, Tuple, Optional, List
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

# --- Helper functions to generate content for specific panels ---
def _generate_header_content(args) -> Panel:
    current_time = time.strftime("%Y-%m-%d %H:%M:%S") #
    # Using args.classifier.value because class_compatible_args will have string
    header_text = f"[bold {THEME.get('title', 'white')}]CIFAR-10 Training with {args.classifier}[/]" #
    date_text = Text.assemble(Text.from_markup(f"[{THEME.get('heading', 'white')}]Date:[/{THEME.get('heading', 'white')}] {current_time}")) #

    header_table = Table.grid(expand=True)
    header_table.add_column(ratio=2)
    header_table.add_column(ratio=1)
    header_table.add_row(Panel(header_text, border_style=THEME.get("accent", "blue")), Align.right(date_text))
    return header_table


def _generate_training_metrics_table(metrics, args, current_epoch, batch_idx, len_train_loader) -> Table:
    avg_loss, avg_acc, last_lr = metrics.get_current_epoch_avg_metrics() #
    if batch_idx is not None and len_train_loader is not None: #
        epoch_info = f"Epoch: {current_epoch}/{args.max_epochs} [{batch_idx}/{len_train_loader}]" #
    else:
        epoch_info = f"Epoch: {current_epoch}/{args.max_epochs}" #

    metrics_table = Table(show_header=True, box=box.SIMPLE, title=epoch_info) #
    metrics_table.add_column("Metric", style=THEME.get("heading", "white")) #
    metrics_table.add_column("Value", style=THEME.get("metrics", "white")) #
    metrics_table.add_row("Current Loss", f"{avg_loss:.4f}") #
    metrics_table.add_row("Current Accuracy", f"{avg_acc*100:.2f}%") #
    metrics_table.add_row("Learning Rate", f"{last_lr:.6f}") #

    if metrics.val_acc_history: #
        best_val_acc = max(metrics.val_acc_history) #
        best_val_epoch = metrics.epochs[metrics.val_acc_history.index(best_val_acc)] #
        metrics_table.add_row("Best Validation Acc", f"[{THEME.get('good', 'green')}]{best_val_acc*100:.2f}%[/{THEME.get('good', 'green')}] (Epoch {best_val_epoch})") #
    return metrics_table

def _generate_model_info_table(args, model) -> Table:
    model_info_table = Table(show_header=True, box=box.SIMPLE, title="Model Information") #
    model_info_table.add_column("Property", style=THEME.get("heading", "white")) #
    model_info_table.add_column("Value", style=THEME.get("metrics", "white")) #
    total_params = sum(p.numel() for p in model.parameters()) #
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) #
    model_info_table.add_row("Architecture", str(args.classifier)) #
    model_info_table.add_row("Total Parameters", f"{total_params:,}") #
    model_info_table.add_row("Trainable Parameters", f"{trainable_params:,}") #
    # ... add other args like batch_size, num_workers etc. from args
    model_info_table.add_row("Batch Size", str(args.batch_size)) #
    model_info_table.add_row("Initial LR", str(args.learning_rate)) #
    return model_info_table

def _generate_noise_renderables(model) -> Tuple[Any, Any]:
    noise_metrics_content = Text("Noise not active or no data.", style="dim") #
    noise_graph_content = Text("Noise not active or no data.", style="dim") #
    noise_title_prefix = "Noise"

    if hasattr(model, 'noise_regularizer') and model.noise_regularizer: #
        noise_title_prefix = model.noise_regularizer.noise_type.capitalize() #
        try:
            noise_metrics_content = model.noise_regularizer.get_metrics_table() #
            noise_graph_content = model.noise_regularizer.create_noise_magnitude_graph(width=80, height=15) #
        except Exception:
            pass # Keep default "No data" text
    return noise_metrics_content, noise_graph_content, noise_title_prefix


def create_dashboard(
        args, # ArgsNamespace for initial setup
        initial_training_metrics, # TrainingMetrics instance
        initial_log_handler,    # LogHandler instance
        model, # For model info and noise regularizer
        epoch_progress_widget: Progress, # Persistent widget
        batch_progress_widget: Progress  # Persistent widget
) -> Tuple[Layout, Dict[str, Any]]:
    """
    Creates the initial dashboard layout and a dictionary of its persistent, updatable components.
    This function is called ONCE.
    """
    layout = Layout(name="root") #
    layout.split_column( #
        Layout(name="header", size=3), #
        Layout(name="body"), #
        Layout(name="footer", size=5)  # Increased footer size for two progress bars
    )
    layout["body"].split_column( #
        Layout(name="graphs_row", ratio=1), #
        Layout(name="info_row", ratio=1) #
    )

    has_noise_regularizer = hasattr(model, 'noise_regularizer') and model.noise_regularizer is not None #

    if not args.disable_graphs:
        graph_row_children = [ #
            Layout(name="accuracy_graph", ratio=1), #
            Layout(name="loss_graph", ratio=1), #
            Layout(name="lr_graph", ratio=1) #
        ]
        if has_noise_regularizer: #
            graph_row_children.insert(2, Layout(name="noise_magnitude_graph", ratio=1)) # Insert before LR graph
        layout["body"]["graphs_row"].split_row(*graph_row_children) #
    else:
        layout["body"]["graphs_row"].update(Panel(Text("ASCII Graphs Disabled", justify="center", style="dim")))
    # Define info row structure
    info_row_children = [ #
        Layout(name="training_metrics_display", ratio=1), #
        Layout(name="model_info_display", ratio=1), #
        Layout(name="log_display", ratio=2) #
    ]
    if has_noise_regularizer: #
        info_row_children.insert(2, Layout(name="noise_metrics_display", ratio=1)) # Insert before log display
    layout["body"]["info_row"].split_row(*info_row_children) #

    # Create and store persistent components/panels
    components = {}

    # Header
    header_table_content = _generate_header_content(args)
    components['header_panel'] = Panel(header_table_content, border_style=THEME.get("accent", "blue")) # No title for the panel itself if the table has one or layout handles it

# Graphs (Panels holding Text content)
    if not args.disable_graphs:
        components['accuracy_graph_content'] = create_accuracy_graph(initial_training_metrics) #
        components['accuracy_graph_panel'] = Panel(components['accuracy_graph_content'], title="Accuracy History", border_style=THEME.get("good", "green"), expand=True) #

        components['loss_graph_content'] = create_loss_graph(initial_training_metrics) #
        components['loss_graph_panel'] = Panel(components['loss_graph_content'], title="Loss History", border_style=THEME.get("bad", "red"), expand=True) #

        components['lr_graph_content'] = create_lr_graph(initial_training_metrics) #
        components['lr_graph_panel'] = Panel(components['lr_graph_content'], title="Learning Rate Schedule", border_style=THEME.get("accent", "blue"), expand=True) #

    if has_noise_regularizer: #
        initial_noise_metrics_content, initial_noise_graph_content, noise_title_prefix = _generate_noise_renderables(model)
        components['noise_metrics_content'] = initial_noise_metrics_content
        components['noise_metrics_panel'] = Panel(components['noise_metrics_content'], title=f"{noise_title_prefix} Noise Metrics", border_style="yellow", expand=True) #
        if not args.disable_graphs:
            components['noise_graph_content'] = initial_noise_graph_content
            components['noise_graph_panel'] = Panel(components['noise_graph_content'], title="Noise Magnitude History", border_style="cyan", expand=True) #

    # Info Panels (Panels holding Tables)
    components['metrics_table_content'] = _generate_training_metrics_table(initial_training_metrics, args, 0, None, None)
    components['metrics_panel'] = Panel(components['metrics_table_content'], title="Training Metrics", border_style=THEME.get("accent", "blue")) #

    components['model_info_table_content'] = _generate_model_info_table(args, model)
    components['model_info_panel'] = Panel(components['model_info_table_content'], title="Model Configuration", border_style=THEME.get("heading", "white")) #

    components['log_table_content'] = initial_log_handler.get_logs_table() #
    components['log_panel'] = Panel(components['log_table_content'], title="Training Log", border_style="green", expand=True) #

    # Store persistent progress widgets
    components['epoch_progress_widget'] = epoch_progress_widget
    components['batch_progress_widget'] = batch_progress_widget

    # Initial population of layout
    layout["header"].update(components['header_panel']) #
    if not args.disable_graphs:
        layout["body"]["graphs_row"]["accuracy_graph"].update(components['accuracy_graph_panel'])
        layout["body"]["graphs_row"]["loss_graph"].update(components['loss_graph_panel'])
        layout["body"]["graphs_row"]["lr_graph"].update(components['lr_graph_panel'])

    layout["body"]["info_row"]["training_metrics_display"].update(components['metrics_panel']) #
    layout["body"]["info_row"]["model_info_display"].update(components['model_info_panel']) #
    layout["body"]["info_row"]["log_display"].update(components['log_panel']) #

    if has_noise_regularizer: #
        if not args.disable_graphs:
            layout["body"]["graphs_row"]["noise_magnitude_graph"].update(components['noise_graph_panel']) #
        layout["body"]["info_row"]["noise_metrics_display"].update(components['noise_metrics_panel']) #

    progress_group = Group(components['batch_progress_widget'], components['epoch_progress_widget']) #
    layout["footer"].update(Panel(progress_group, title="Progress")) # (Updated to Panel)

    return layout, components

def update_dashboard_contents(
        components: Dict[str, Any],
        training_metrics, # TrainingMetrics instance
        log_handler,      # LogHandler instance
        args: Any,        # ArgsNamespace instance
        model: Any,       # CIFAR10Module instance
        current_epoch: int,
        max_epochs: int, # Used by some content generators
        batch_idx: Optional[int] = None,
        len_train_loader: Optional[int] = None
):
    """
    Updates the content of the persistent dashboard components.
    Progress bars are updated externally.
    """
    # Update Header (time might change, or other dynamic info)
    components['header_panel'].renderable = _generate_header_content(args) # Assuming _generate_header_content returns a Panel or compatible table for update

    # Update Graph Panels by replacing their Text content
    if not args.disable_graphs:
        components['accuracy_graph_panel'].renderable = create_accuracy_graph(training_metrics) #
        components['loss_graph_panel'].renderable = create_loss_graph(training_metrics) #
        components['lr_graph_panel'].renderable = create_lr_graph(training_metrics) #

    # Update Info Panels by replacing their Table content
    components['metrics_panel'].renderable = _generate_training_metrics_table(training_metrics, args, current_epoch, batch_idx, len_train_loader) #
    # Model info typically doesn't change, but if it could:
    # components['model_info_panel'].renderable = _generate_model_info_table(args, model)

    components['log_panel'].renderable = log_handler.get_logs_table() #

    if hasattr(model, 'noise_regularizer') and model.noise_regularizer: #
        if 'noise_metrics_panel' in components and 'noise_graph_panel' in components:
            noise_metrics_content, noise_graph_content, noise_title_prefix = _generate_noise_renderables(model)
            components['noise_metrics_panel'].renderable = noise_metrics_content #
            components['noise_metrics_panel'].title = f"{noise_title_prefix} Noise Metrics" #
            if not args.disable_graphs:
                components['noise_graph_panel'].renderable = noise_graph_content #


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
