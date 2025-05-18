from enum import Enum

import torch
import torch.nn as nn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Add current directory to path to ensure imports work properly
import sys
import os
sys.path.append(os.getcwd())

from .cifar10_models.densenet import densenet121, densenet161, densenet169
from .cifar10_models.googlenet import googlenet
from .cifar10_models.inception import inception_v3
from .cifar10_models.mobilenetv2 import mobilenet_v2
from .cifar10_models.resnet import resnet18, resnet34, resnet50
from .cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from .schduler import WarmupCosineLR

# Initialize console for rich output
console = Console()

# Check if DirectML is available
# try:
#     import torch_directml
#     HAS_DIRECTML = True
# except (ImportError, TypeError) as e:
#     console.print(Panel(f"[yellow]DirectML import error: {e}[/yellow]",
#                         title="Import Warning", border_style="yellow"))
HAS_DIRECTML = False

all_classifiers = {
    "vgg11_bn": vgg11_bn(),
    "vgg13_bn": vgg13_bn(),
    "vgg16_bn": vgg16_bn(),
    "vgg19_bn": vgg19_bn(),
    "resnet18": resnet18(),
    "resnet34": resnet34(),
    "resnet50": resnet50(),
    "densenet121": densenet121(),
    "densenet161": densenet161(),
    "densenet169": densenet169(),
    "mobilenet_v2": mobilenet_v2(),
    "googlenet": googlenet(),
    "inception_v3": inception_v3(),
}

# Architecture information
architecture_info = {
    "vgg11_bn": {"type": "VGG", "layers": 11, "params": "132.9M", "description": "VGG-11 with batch normalization"},
    "vgg13_bn": {"type": "VGG", "layers": 13, "params": "133.1M", "description": "VGG-13 with batch normalization"},
    "vgg16_bn": {"type": "VGG", "layers": 16, "params": "138.4M", "description": "VGG-16 with batch normalization"},
    "vgg19_bn": {"type": "VGG", "layers": 19, "params": "143.7M", "description": "VGG-19 with batch normalization"},
    "resnet18": {"type": "ResNet", "layers": 18, "params": "11.7M", "description": "ResNet with 18 layers"},
    "resnet34": {"type": "ResNet", "layers": 34, "params": "21.8M", "description": "ResNet with 34 layers"},
    "resnet50": {"type": "ResNet", "layers": 50, "params": "25.6M", "description": "ResNet with 50 layers"},
    "densenet121": {"type": "DenseNet", "layers": 121, "params": "8.0M", "description": "DenseNet with 121 layers"},
    "densenet161": {"type": "DenseNet", "layers": 161, "params": "28.7M", "description": "DenseNet with 161 layers"},
    "densenet169": {"type": "DenseNet", "layers": 169, "params": "14.2M", "description": "DenseNet with 169 layers"},
    "mobilenet_v2": {"type": "MobileNet", "layers": "-", "params": "3.5M", "description": "MobileNetV2"},
    "googlenet": {"type": "GoogleNet", "layers": "-", "params": "6.8M", "description": "GoogLeNet/Inception v1"},
    "inception_v3": {"type": "Inception", "layers": "-", "params": "27.2M", "description": "Inception v3"},
}

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

class PerturbedCIFAR10Module(nn.Module):
    def __init__(self, args):
        super().__init__()

        #print("CLASSIFIER ARGUMENT", args.classifier)

        assert args.noise_type in [None, 'gaussian', 'uniform'], "Invalid noise type"
        assert args.noise_position in [None, 'input', 'weight', 'gradient'], "Invalid noise position"
        #assert args.classifier in all_classifiers.keys(), f"Invalid classifier: {args.classifier}"

        # Store arguments
        self.args = args

        # Initialize model, criterion, and accuracy tracker
        self.criterion = nn.CrossEntropyLoss()

        # Get model architecture with nice log formatting
        console.print(Panel(f"[bold]Loading model architecture:[/bold] [cyan]{args.classifier}[/cyan]",
                            title="Model Initialization", border_style="blue"))

        if args.classifier in architecture_info:
            arch_info = architecture_info[args.classifier]
            info_table = Table(box=box.SIMPLE, title=f"{args.classifier} Architecture")
            info_table.add_column("Property", style="yellow")
            info_table.add_column("Value", style="cyan")

            info_table.add_row("Type", arch_info["type"])
            info_table.add_row("Layers", str(arch_info["layers"]))
            info_table.add_row("Parameters (approx)", arch_info["params"])
            info_table.add_row("Description", arch_info["description"])

            console.print(info_table)

        self.model = all_classifiers[args.classifier]

        # Print model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        console.print(f"[bold]Model parameters:[/bold] {total_params:,} total, {trainable_params:,} trainable")

        #print("NOISE PARAMETERS:", args.noise_layer)

        if not args.noise_layer:
            self.perturbed_layers_names = [name for name, _ in self.model.named_parameters()]
        else:
            self.perturbed_layers_names = args.noise_layer

        self.param_dict = dict(self.model.named_parameters())

        # Print model layers summary (simplified)
        self._print_model_summary()

        # Check if DirectML device should be used
        if HAS_DIRECTML and hasattr(args, 'gpu_id') and args.gpu_id != "-1":
            # Check for cpu_only flag
            use_cpu = False
            if hasattr(args, 'cpu_only'):
                use_cpu = bool(args.cpu_only)

            # Only set directml_available if all conditions are met
            self.directml_available = not use_cpu
        else:
            self.directml_available = False

# ---------------------------------- PERTURBATION PART OF THE NN ----------------------------------

    # Define the different types of noise
    def getNoise(self, x):
        if self.args.noise_type is None:
            return torch.zeros_like(x)
        elif self.args.noise_type == 'gaussian':
            return torch.randn_like(x) * self.args.noise_std
        elif self.args.noise_type == 'uniform':
            a = (3**0.5) * self.args.noise_std # uniform distribution with requested std
            return torch.empty_like(x).uniform_(-a, a)
        else:
            raise ValueError("Invalid noise type. Choose 'gaussian' or 'uniform'.")

    def forward(self, x):
        # Apply input noise
        if self.args.noise_position == "input":
            x = self.apply_input_noise(x)

        # Apply weight noise before forward
        if self.args.noise_position == "weight":
            self.save_original_weights()
            with torch.no_grad():
                self.apply_weight_noise()

        # Register gradient noise hook once (only first time)
        elif self.args.noise_position == "gradient":
            self.register_gradient_noise_hook()

        # Forward pass
        out = self.model(x)

        # Restore weights after forward if needed
        if self.args.noise_position == "weight":
            with torch.no_grad():
                self.restore_weights()

        return out

    def apply_input_noise(self, x):
        noise = self.getNoise(x)
        return x + noise

    def apply_weight_noise(self):
        # Assuming you want to apply noise to the model's weights.
        for name, param in self.model.named_parameters():
            if param.requires_grad and any(layer in name for layer in self.perturbed_layers_names):
                print("Add Perturbation to layer: ", name)
                noise = self.getNoise(param)
                param.data += noise

    def save_original_weights(self):
        # Save the original weights before adding noise, so they can be restored later.
        self.original_weights = {name: param.clone() for name, param in self.model.named_parameters() if
                                 param.requires_grad}

    def restore_weights(self):
        # Restore the original weights after the forward pass.
        for name, param in self.model.named_parameters():
            if (param.requires_grad and
                    name in self.original_weights and
                    any(layer in name for layer in self.perturbed_layers_names)):
                print("Restore Perturbation to layer: ", name)
                param.data = self.original_weights[name].data

    def register_gradient_noise_hook(self):
        # Function called during backpropagation to add noise to gradients
        def add_noise_to_grad(grad):
            return grad + self.getNoise(grad)

        for name, param in self.model.named_parameters():
            if (param.requires_grad and
                any(layer in name for layer in self.perturbed_layers_names)):
                print("Add Perturbation at layer: ", name)
                # Register the hook to add noise to gradients
                param.register_hook(add_noise_to_grad)

# -------------------------------------------------------------------------------------------------

    def _print_model_summary(self):
        """Print a simplified summary of the model layers"""
        # Count layer types
        layer_counts = {}
        for name, module in self.model.named_modules():
            layer_type = module.__class__.__name__
            if layer_type not in ["Sequential", "ModuleList", "ModuleDict", "BasicBlock", "Bottleneck", "ResNet"]:
                if layer_type not in layer_counts:
                    layer_counts[layer_type] = 0
                layer_counts[layer_type] += 1

        # Create layer summary table
        layer_table = Table(box=box.SIMPLE, title="Layer Summary")
        layer_table.add_column("Layer Type", style="yellow")
        layer_table.add_column("Count", justify="right", style="cyan")

        for layer_type, count in sorted(layer_counts.items(), key=lambda x: x[1], reverse=True):
            layer_table.add_row(layer_type, str(count))

        console.print(layer_table)

    def configure_optimizer(self):
        # Create optimizer with momentum
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            momentum=0.9,
            nesterov=True,
        )

        # Calculate total steps for the scheduler (approximate for CIFAR-10)
        total_steps = self.args.max_epochs * 50000 // self.args.batch_size
        warmup_steps = int(total_steps * 0.3)

        # Create learning rate scheduler with warmup
        scheduler = WarmupCosineLR(
            optimizer,
            warmup_epochs=warmup_steps,
            max_epochs=total_steps,
            warmup_start_lr=1e-6,  # Starting learning rate for warmup
            eta_min=1e-6,          # Minimum learning rate
        )

        # Log optimizer and scheduler setup
        optim_table = Table(box=box.SIMPLE, title="Optimization Setup")
        optim_table.add_column("Parameter", style="yellow")
        optim_table.add_column("Value", style="cyan")

        optim_table.add_row("Optimizer", "SGD")
        optim_table.add_row("Learning Rate", str(self.args.learning_rate))
        optim_table.add_row("Weight Decay", str(self.args.weight_decay))
        optim_table.add_row("Momentum", "0.9")
        optim_table.add_row("Nesterov", "True")
        optim_table.add_row("Scheduler", "WarmupCosineLR")
        optim_table.add_row("Warmup Steps", str(warmup_steps))
        optim_table.add_row("Total Steps", str(total_steps))
        optim_table.add_row("Min LR", "1e-6")

        console.print(Panel(optim_table, title="Optimizer Configuration", border_style="green"))

        return optimizer, scheduler

    def calculate_accuracy(self, outputs, targets):
        """Calculate accuracy from model outputs and targets"""
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        total = targets.size(0)
        return correct, total