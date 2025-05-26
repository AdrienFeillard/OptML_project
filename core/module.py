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
from .cifar10_models.SimpleCNN import simple_cnn, tiny_cnn
from .schduler import WarmupCosineLR
from .noise_regularization import NoiseType, NoiseRegularizer, NoiseDistribution

# Initialize console for rich output
console = Console()

# Check if DirectML is available
try:
    import torch_directml
    HAS_DIRECTML = True
except (ImportError, TypeError) as e:
    #console.print(Panel(f"[yellow]DirectML import error: {e}[/yellow]", title="Import Warning", border_style="yellow"))
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
    "simple_cnn": simple_cnn(),
    "tiny_cnn": tiny_cnn()
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

class Optimizer(str, Enum):
    SGD = "sgd"
    Adam = "adam"

class CIFAR10Module(nn.Module):
    def __init__(self, args):
        super().__init__()
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

        # Initialize noise regularizer if specified
        self.noise_regularizer = None
        if hasattr(args, 'noise_type') and args.noise_type != NoiseType.none:
            # Get noise distribution (default to gaussian if not specified)
            noise_distribution = getattr(args, 'noise_distribution', NoiseDistribution.gaussian)

            console.print(Panel(
                f"[bold]Setting up {args.noise_type} noise regularization with {noise_distribution} distribution[/bold]",
                title="Noise Regularization",
                border_style="yellow"
            ))

            self.noise_regularizer = NoiseRegularizer(
                noise_type=args.noise_type,
                magnitude=args.noise_magnitude,
                schedule=args.noise_schedule,
                max_epochs=args.max_epochs,
                apply_to_layers=args.noise_layer,
                noise_distribution=noise_distribution
            )

            # Register gradient hooks if using gradient noise
            if args.noise_type == NoiseType.gradient:
                self.noise_regularizer.register_gradient_noise_hook(self)

        # Print model layers summary
        self._print_model_summary()

        # Check for DirectML device
        if HAS_DIRECTML and hasattr(args, 'gpu_id') and args.gpu_id != "-1":
            # Check for cpu_only flag
            use_cpu = False
            if hasattr(args, 'cpu_only'):
                use_cpu = bool(args.cpu_only)

            # Only set directml_available if all conditions are met
            self.directml_available = not use_cpu
        else:
            self.directml_available = False

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

    def forward(self, images):
        # Apply input noise if configured
        if hasattr(self, 'noise_regularizer') and self.noise_regularizer and self.noise_regularizer.noise_type == NoiseType.input:
            images = self.noise_regularizer.apply_input_noise(images)

        # Apply weight noise if configured (temporary during forward pass)
        if hasattr(self, 'noise_regularizer') and self.noise_regularizer and self.noise_regularizer.noise_type == NoiseType.weight:
            self.noise_regularizer.save_original_weights(self)
            self.noise_regularizer.apply_weight_noise(self, permanent=False)

        # Forward pass through the model
        outputs = self.model(images)

        # Restore original weights if we applied weight noise
        if hasattr(self, 'noise_regularizer') and self.noise_regularizer and self.noise_regularizer.noise_type == NoiseType.weight:
            self.noise_regularizer.restore_weights(self)

        return outputs

    def configure_optimizer(self):
        # Create optimizer with momentum

        if self.args.optimizer == "sgd":
            print("Using SGD optimizer")
            print("momentum: ", self.args.momentum)
            print("weight_decay: ", self.args.weight_decay)
            print("learning_rate: ", self.args.learning_rate)

            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                momentum=self.args.momentum,
                nesterov=(self.args.momentum > 0),
            )
        elif self.args.optimizer == "adam":

            print("Using Adam optimizer")
            print("learning_rate: ", self.args.learning_rate)
            print("weight_decay: ", self.args.weight_decay)
            print("beta1: ", self.args.beta1)
            print("beta2: ", self.args.beta2)

            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                betas=(self.args.beta1, self.args.beta2),

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