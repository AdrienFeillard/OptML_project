import torch
import torch.nn as nn
import numpy as np
from enum import Enum
from typing import Optional, List, Dict, Union
import typer
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
import matplotlib.pyplot as plt
import os
import time

# Define noise types as Enum for CLI
class NoiseType(str, Enum):
    none = "none"
    gradient = "gradient"
    weight = "weight"
    input = "input"
    label = "label"
    dropout = "dropout"

class NoiseDistribution(str, Enum):
    gaussian = "gaussian"
    uniform = "uniform"

class NoiseSchedule(str, Enum):
    constant = "constant"
    linear = "linear"
    cosine = "cosine"
    exponential = "exponential"

class NoiseRegularizer:
    """Class for applying various noise regularization techniques"""
    def __init__(
            self,
            noise_type: NoiseType,
            magnitude: float,
            max_epochs: int,
            schedule: NoiseSchedule = NoiseSchedule.constant,
            # Layer-specific parameters
            apply_to_layers: Optional[List[str]] = None,
            # Dropout specific parameters
            dropout_prob: float = 0.1,
            # Noise distribution
            noise_distribution: NoiseDistribution = NoiseDistribution.gaussian,
            # Tracking metrics
            track_metrics: bool = True
    ):
        self.noise_type = noise_type
        self.initial_magnitude = magnitude
        self.current_magnitude = magnitude
        self.schedule = schedule
        self.max_epochs = max_epochs
        self.current_epoch = 0
        self.apply_to_layers = apply_to_layers
        self.dropout_prob = dropout_prob
        self.noise_distribution = noise_distribution
        self.track_metrics = track_metrics

        # Initialize metrics tracking
        self.metrics = {
            'gradient_norm_before': [],
            'gradient_norm_after': [],
            'weight_norm_before': [],
            'weight_norm_after': [],
            'input_norm_before': [],
            'input_norm_after': [],
            'noise_magnitude': [],
            'epoch': []
        }

        # For layer-specific application
        self.layer_metrics = {}

        # For weight noise preservation
        self.original_weights = {}
        self.has_registered_hooks = False

    def update_epoch(self, epoch: int):
        """Update current epoch for noise scheduling"""
        self.current_epoch = epoch
        self._update_magnitude()

    def _update_magnitude(self):
        """Update noise magnitude based on schedule"""
        if self.schedule == NoiseSchedule.constant:
            self.current_magnitude = self.initial_magnitude
        elif self.schedule == NoiseSchedule.linear:
            # Linear decay from initial_magnitude to 0
            progress = min(1.0, self.current_epoch / self.max_epochs)
            self.current_magnitude = self.initial_magnitude * (1 - progress)
        elif self.schedule == NoiseSchedule.cosine:
            # Cosine annealing
            progress = min(1.0, self.current_epoch / self.max_epochs)
            self.current_magnitude = self.initial_magnitude * 0.5 * (1 + np.cos(np.pi * progress))
        elif self.schedule == NoiseSchedule.exponential:
            # Exponential decay
            decay_rate = 0.95
            self.current_magnitude = self.initial_magnitude * (decay_rate ** self.current_epoch)

        # Record the current magnitude
        self.metrics['noise_magnitude'].append(self.current_magnitude)
        self.metrics['epoch'].append(self.current_epoch)

    def get_noise(self, x):
        """Generate noise based on the specified distribution"""
        if self.noise_distribution == NoiseDistribution.gaussian:
            return torch.randn_like(x) * self.current_magnitude
        elif self.noise_distribution == NoiseDistribution.uniform:
            # Scale uniform noise to match the variance of gaussian
            a = (3**0.5) * self.current_magnitude  # uniform distribution with requested std
            return torch.empty_like(x).uniform_(-a, a)
        else:
            raise ValueError(f"Invalid noise distribution: {self.noise_distribution}")

    def apply_gradient_noise(self, model: nn.Module):
        """Apply noise to gradients during backpropagation"""
        total_norm_before = 0
        total_norm_after = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                # Check if we should apply noise to this layer
                if self.apply_to_layers and not any(layer_name in name for layer_name in self.apply_to_layers):
                    continue

                # Track gradient norm before noise
                grad_norm_before = param.grad.norm().item()
                total_norm_before += grad_norm_before

                # Apply noise to gradients based on distribution
                noise = self.get_noise(param.grad)
                param.grad.add_(noise)

                # Track gradient norm after noise
                grad_norm_after = param.grad.norm().item()
                total_norm_after += grad_norm_after

                # Track per-layer metrics
                if name not in self.layer_metrics:
                    self.layer_metrics[name] = {
                        'grad_norm_before': [],
                        'grad_norm_after': [],
                        'noise_effect': []  # Ratio of after/before
                    }

                self.layer_metrics[name]['grad_norm_before'].append(grad_norm_before)
                self.layer_metrics[name]['grad_norm_after'].append(grad_norm_after)
                self.layer_metrics[name]['noise_effect'].append(grad_norm_after / max(1e-8, grad_norm_before))

        # Record overall metrics
        if self.track_metrics:
            self.metrics['gradient_norm_before'].append(total_norm_before)
            self.metrics['gradient_norm_after'].append(total_norm_after)

    def register_gradient_noise_hook(self, model: nn.Module):
        """Register hooks to add noise to gradients during backpropagation"""
        if self.has_registered_hooks:
            return

        total_hooks = 0

        # Create hook for adding noise to gradients
        def add_noise_to_grad(name):
            def hook(grad):
                # Track grad norm before
                grad_norm_before = grad.norm().item()

                # Add noise
                noisy_grad = grad + self.get_noise(grad)

                # Track grad norm after
                grad_norm_after = noisy_grad.norm().item()

                # Add metrics to layer data
                if name in self.layer_metrics:
                    self.layer_metrics[name].setdefault('grad_norm_before', []).append(grad_norm_before)
                    self.layer_metrics[name].setdefault('grad_norm_after', []).append(grad_norm_after)
                    self.layer_metrics[name].setdefault('noise_effect', []).append(grad_norm_after / max(1e-8, grad_norm_before))

                return noisy_grad
            return hook

        # Register hooks for each parameter
        for name, param in model.named_parameters():
            if (param.requires_grad and
                    (not self.apply_to_layers or any(layer_name in name for layer_name in self.apply_to_layers))):

                print("GRAD HOOK REGISTERED FOR", name)
                param.register_hook(add_noise_to_grad(name))
                total_hooks += 1

        self.has_registered_hooks = True
        print(f"Added {total_hooks} gradient noise hooks")

    def apply_weight_noise(self, model: nn.Module, permanent: bool = False):
        """Apply noise directly to model weights"""
        total_norm_before = 0
        total_norm_after = 0

        # Context manager for temporary weight noise
        context = torch.no_grad() if permanent else torch.enable_grad()

        with context:
            for name, param in model.named_parameters():
                # Check if we should apply noise to this layer
                if self.apply_to_layers and not any(layer_name in name for layer_name in self.apply_to_layers):
                    continue

                # Track weight norm before noise
                weight_norm_before = param.norm().item()
                total_norm_before += weight_norm_before

                # Apply noise to weights - AVOID IN-PLACE OPERATIONS by using clone and assignment
                if param.requires_grad and not permanent:
                    # For params that require gradients, create a new tensor
                    noise = self.get_noise(param)
                    param.data = param.data + noise  # Use .data to modify the tensor's data without triggering autograd
                else:
                    # For params that don't require gradients or if applying permanently
                    noise = self.get_noise(param)
                    param.add_(noise)  # In-place is fine for this case

                # Track weight norm after noise
                weight_norm_after = param.norm().item()
                total_norm_after += weight_norm_after

                # Track per-layer metrics
                if name not in self.layer_metrics:
                    self.layer_metrics[name] = {
                        'weight_norm_before': [],
                        'weight_norm_after': [],
                        'noise_effect': []  # Ratio of after/before
                    }

                self.layer_metrics[name]['weight_norm_before'].append(weight_norm_before)
                self.layer_metrics[name]['weight_norm_after'].append(weight_norm_after)
                self.layer_metrics[name]['noise_effect'].append(weight_norm_after / max(1e-8, weight_norm_before))

        # Record overall metrics
        if self.track_metrics:
            self.metrics['weight_norm_before'].append(total_norm_before)
            self.metrics['weight_norm_after'].append(total_norm_after)

    def save_original_weights(self, model: nn.Module):
        """Save original weights before perturbation"""
        self.original_weights = {
            name: param.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def restore_weights(self, model: nn.Module):
        """Restore original weights after forward pass"""
        if not self.original_weights:
            return

        with torch.no_grad():
            for name, param in model.named_parameters():
                if (param.requires_grad and
                        name in self.original_weights and
                        (not self.apply_to_layers or any(layer_name in name for layer_name in self.apply_to_layers))):
                    param.data = self.original_weights[name].data

        # Clear the original weights after restoring
        self.original_weights = {}

    def apply_input_noise(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply noise to input data"""
        # Track input norm before
        if self.track_metrics:
            input_norm_before = inputs.norm().item()
            self.metrics['input_norm_before'].append(input_norm_before)

        # Apply noise based on distribution
        noise = self.get_noise(inputs)
        noisy_inputs = inputs + noise

        # Track input norm after
        if self.track_metrics:
            input_norm_after = noisy_inputs.norm().item()
            self.metrics['input_norm_after'].append(input_norm_after)

        return noisy_inputs

    def apply_label_noise(self, labels: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
        """Apply noise to labels by randomly flipping with probability based on magnitude"""
        # Current magnitude controls probability of label flip
        flip_prob = min(0.5, self.current_magnitude)  # Cap at 50% to avoid too much chaos

        # Create a mask of labels to flip
        flip_mask = torch.rand_like(labels.float()) < flip_prob

        # For masked elements, generate random new labels
        random_labels = torch.randint_like(labels, 0, num_classes)

        # Where mask is True, use random label, else keep original
        noisy_labels = torch.where(flip_mask, random_labels, labels)

        # Record metrics for label noise if needed
        if self.track_metrics:
            self.metrics.setdefault('labels_flipped', []).append(flip_mask.sum().item() / labels.numel())

        return noisy_labels

    def get_metrics_table(self) -> Table:
        """Create a Rich table with current metrics"""
        table = Table(title="Noise Regularization Metrics", box=box.SIMPLE)

        table.add_column("Metric", style="yellow")
        table.add_column("Value", style="cyan")

        # Add noise settings
        table.add_row("Noise Type", f"{self.noise_type}")
        table.add_row("Noise Distribution", f"{self.noise_distribution}")
        table.add_row("Current Magnitude", f"{self.current_magnitude:.6f}")
        table.add_row("Schedule", f"{self.schedule}")

        # Add specific metrics based on noise type
        if self.noise_type == NoiseType.gradient and self.metrics['gradient_norm_before']:
            last_idx = -1
            before = self.metrics['gradient_norm_before'][last_idx]
            after = self.metrics['gradient_norm_after'][last_idx]
            table.add_row("Gradient Norm (Before)", f"{before:.4f}")
            table.add_row("Gradient Norm (After)", f"{after:.4f}")
            table.add_row("Norm Ratio", f"{after/before:.4f}")

        elif self.noise_type == NoiseType.weight and self.metrics['weight_norm_before']:
            last_idx = -1
            before = self.metrics['weight_norm_before'][last_idx]
            after = self.metrics['weight_norm_after'][last_idx]
            table.add_row("Weight Norm (Before)", f"{before:.4f}")
            table.add_row("Weight Norm (After)", f"{after:.4f}")
            table.add_row("Norm Ratio", f"{after/before:.4f}")

        elif self.noise_type == NoiseType.input and self.metrics['input_norm_before']:
            last_idx = -1
            before = self.metrics['input_norm_before'][last_idx]
            after = self.metrics['input_norm_after'][last_idx]
            table.add_row("Input Norm (Before)", f"{before:.4f}")
            table.add_row("Input Norm (After)", f"{after:.4f}")
            table.add_row("Norm Ratio", f"{after/before:.4f}")

        elif self.noise_type == NoiseType.label and 'labels_flipped' in self.metrics:
            last_idx = -1
            flipped = self.metrics['labels_flipped'][last_idx]
            table.add_row("Labels Flipped", f"{flipped:.2%}")

        return table

    def create_noise_magnitude_graph(self, width=60, height=10):
        """Create a simple ASCII graph of noise magnitude history"""
        if not self.metrics['noise_magnitude']:
            return Text("No data yet", style="dim")

        # Setup the canvas
        canvas = [[" " for _ in range(width)] for _ in range(height)]

        # Get data
        magnitudes = self.metrics['noise_magnitude']
        epochs = self.metrics['epoch']

        # Calculate scaling factors
        y_max = max(magnitudes) * 1.1
        y_min = 0

        x_max = max(epochs)
        x_min = 0

        # Scale factors
        x_scale = (width - 1) / max(1, (x_max - x_min))
        y_scale = (height - 1) / max(0.0001, (y_max - y_min))

        # Plot points
        for i in range(len(magnitudes)):
            x = int((epochs[i] - x_min) * x_scale)
            y = height - 1 - int((magnitudes[i] - y_min) * y_scale)

            # Ensure within bounds
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))

            # Plot point
            canvas[y][x] = "◉"

        # Convert canvas to text
        result = ""
        for row in canvas:
            result += "".join(row) + "\n"

        # Add axis
        x_axis = "-" * width

        # Create the graph
        graph = Text.assemble(
            Text(result, style="cyan"),
            Text(f"\n{x_axis}", style="blue"),
            Text.from_markup(f"\nNoise Magnitude: [bold cyan]{self.current_magnitude:.6f}[/bold cyan]")
        )

        return graph

    def plot_metrics_to_file(self, filename: str = "noise_metrics.png"):
        """Save metrics plots to file for later analysis"""
        if not self.metrics['epoch']:
            return "No data to plot"

        # Create the plot
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        # Plot noise magnitude over time
        axs[0].plot(self.metrics['epoch'], self.metrics['noise_magnitude'], 'b-', label='Noise Magnitude')
        axs[0].set_title('Noise Magnitude over Epochs')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Magnitude')
        axs[0].legend()
        axs[0].grid(True)

        # Plot specific metrics based on noise type
        if self.noise_type == NoiseType.gradient and self.metrics['gradient_norm_before']:
            axs[1].plot(self.metrics['epoch'], self.metrics['gradient_norm_before'], 'g-', label='Before Noise')
            axs[1].plot(self.metrics['epoch'], self.metrics['gradient_norm_after'], 'r-', label='After Noise')
            axs[1].set_title('Gradient Norm Before/After Noise')

        elif self.noise_type == NoiseType.weight and self.metrics['weight_norm_before']:
            axs[1].plot(self.metrics['epoch'], self.metrics['weight_norm_before'], 'g-', label='Before Noise')
            axs[1].plot(self.metrics['epoch'], self.metrics['weight_norm_after'], 'r-', label='After Noise')
            axs[1].set_title('Weight Norm Before/After Noise')

        elif self.noise_type == NoiseType.input and self.metrics['input_norm_before']:
            axs[1].plot(self.metrics['epoch'], self.metrics['input_norm_before'], 'g-', label='Before Noise')
            axs[1].plot(self.metrics['epoch'], self.metrics['input_norm_after'], 'r-', label='After Noise')
            axs[1].set_title('Input Norm Before/After Noise')

        elif self.noise_type == NoiseType.label and 'labels_flipped' in self.metrics:
            axs[1].plot(self.metrics['epoch'], self.metrics['labels_flipped'], 'r-', label='Labels Flipped')
            axs[1].set_title('Percentage of Labels Flipped')

        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Value')
        axs[1].legend()
        axs[1].grid(True)

        # Save the plot
        os.makedirs('plots', exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join('plots', filename))
        plt.close()

        return f"Metrics saved to {os.path.join('plots', filename)}"

    def create_layer_sensitivity_heatmap(self, filename: str = "layer_sensitivity.png"):
        """Create and save a heatmap showing which layers are most affected by noise"""
        if not self.layer_metrics:
            return "No layer-specific data collected"

        # Prepare data for heatmap
        layer_names = []
        noise_effects = []

        for name, metrics in self.layer_metrics.items():
            if 'noise_effect' in metrics and metrics['noise_effect']:
                # Use last 5 epochs to get a stable measure
                avg_effect = np.mean(metrics['noise_effect'][-5:])
                layer_names.append(name.split('.')[-1][:10])  # Truncate to prevent long names
                noise_effects.append(avg_effect)

        if not layer_names:
            return "No layer effect data available"

        # Sort by effect magnitude
        sorted_indices = np.argsort(noise_effects)
        layer_names = [layer_names[i] for i in sorted_indices]
        noise_effects = [noise_effects[i] for i in sorted_indices]

        # Create plot
        fig, ax = plt.figure(figsize=(12, len(layer_names) * 0.4 + 2)), plt.gca()

        # Create horizontal bar chart
        bars = ax.barh(layer_names, noise_effects, color='skyblue')

        # Add a red line at ratio = 1.0 (no effect)
        ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)

        # Color bars based on effect (green for stabilizing, red for destabilizing)
        for i, bar in enumerate(bars):
            if noise_effects[i] < 0.95:  # Stabilizing effect
                bar.set_color('green')
            elif noise_effects[i] > 1.05:  # Destabilizing effect
                bar.set_color('red')

        ax.set_title(f'Layer Sensitivity to {self.noise_type.capitalize()} Noise')
        ax.set_xlabel('Effect Ratio (>1 means amplification)')

        # Save the plot
        os.makedirs('plots', exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join('plots', filename))
        plt.close()

        return f"Layer sensitivity heatmap saved to {os.path.join('plots', filename)}"