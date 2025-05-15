import torch
import json
import os
import time
from datetime import datetime

class TrainingMetrics:
    """Class to track and store training metrics over time"""
    def __init__(self, experiment_name=None, save_dir="checkpoints/metrics", max_history=100):
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

        # For saving metrics
        self.experiment_name = experiment_name
        if experiment_name:
            # Add timestamp to make experiment name unique
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{experiment_name}_{timestamp}"

            # Create metrics directory
            self.metrics_dir = os.path.join(save_dir, self.experiment_name)
            os.makedirs(self.metrics_dir, exist_ok=True)

            print(f"Metrics will be saved to: {self.metrics_dir}")
        else:
            self.metrics_dir = None

        # For optimization metrics
        self.gradient_norms = []
        self.weight_norms = []
        self.noise_metrics = []
        self.sharpness_metrics = []
        self.config = {}
        self.final_metrics = {}

        # Record start time
        self.start_time = time.time()

        # Save start timestamp if metrics directory exists
        if self.metrics_dir:
            self._save_timestamp("start")

    def set_config(self, args):
        """Save experiment configuration"""
        # Extract serializable attributes from args object
        for key, value in vars(args).items():
            # Handle Enum types and other non-serializable types
            if hasattr(value, 'value'):  # For Enum types
                self.config[key] = value.value
            else:
                try:
                    # Try to serialize to JSON to check if it's serializable
                    json.dumps(value)
                    self.config[key] = value
                except (TypeError, OverflowError):
                    # If not serializable, convert to string
                    self.config[key] = str(value)

        # Save config if metrics directory exists
        if self.metrics_dir:
            self._save_metrics("config.json", self.config)

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

        # Save metrics if directory exists
        if self.metrics_dir:
            epoch_data = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": lr,
                "timestamp": time.time()
            }

            # Update epoch metrics list
            epoch_metrics = self._load_metrics("epoch_metrics.json", [])
            epoch_metrics.append(epoch_data)
            self._save_metrics("epoch_metrics.json", epoch_metrics)

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

    def log_gradient_norms(self, model, epoch, batch_idx=None):
        """Log gradient norms for optimization analysis"""
        if not self.metrics_dir:
            return None

        grad_norms = {}
        total_norm = 0.0

        # Collect norms by layer
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                grad_norms[name] = param_norm
                total_norm += param_norm ** 2

        # Calculate total norm
        total_norm = total_norm ** 0.5

        # Create norm data
        norm_data = {
            "epoch": epoch,
            "batch": batch_idx,
            "total_norm": total_norm,
            "layer_norms": grad_norms,
            "timestamp": time.time()
        }

        # Add to gradient norms list
        self.gradient_norms.append(norm_data)

        # Save optimization metrics
        if batch_idx is None or batch_idx % 50 == 0:
            optim_metrics = {
                "gradient_norms": self.gradient_norms,
                "weight_norms": self.weight_norms,
                "sharpness_metrics": self.sharpness_metrics
            }
            self._save_metrics("optimization_metrics.json", optim_metrics)

        return total_norm

    def log_weight_norms(self, model, epoch, batch_idx=None):
        """Log weight norms for optimization analysis"""
        if not self.metrics_dir:
            return None

        weight_norms = {}
        total_norm = 0.0

        # Collect norms by layer
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_norm = param.data.norm(2).item()
                weight_norms[name] = param_norm
                total_norm += param_norm ** 2

        # Calculate total norm
        total_norm = total_norm ** 0.5

        # Create norm data
        norm_data = {
            "epoch": epoch,
            "batch": batch_idx,
            "total_norm": total_norm,
            "layer_norms": weight_norms,
            "timestamp": time.time()
        }

        # Add to weight norms list
        self.weight_norms.append(norm_data)

        # Save optimization metrics
        if batch_idx is None or batch_idx % 50 == 0:
            optim_metrics = {
                "gradient_norms": self.gradient_norms,
                "weight_norms": self.weight_norms,
                "sharpness_metrics": self.sharpness_metrics
            }
            self._save_metrics("optimization_metrics.json", optim_metrics)

        return total_norm

    def log_noise_metrics(self, noise_type, noise_magnitude, norms_before=None, norms_after=None, epoch=None):
        """Log noise-related metrics"""
        if not self.metrics_dir:
            return

        noise_data = {
            "epoch": epoch,
            "noise_type": noise_type,
            "magnitude": noise_magnitude,
            "norms_before": norms_before,
            "norms_after": norms_after,
            "timestamp": time.time()
        }

        # Add to noise metrics list
        self.noise_metrics.append(noise_data)

        # Save noise metrics
        self._save_metrics("noise_metrics.json", self.noise_metrics)

    def measure_sharpness(self, model, criterion, val_loader, device, samples=10, sigma=0.01):
        """Measure sharpness of found minimum by perturbation sensitivity"""
        if not self.metrics_dir:
            return None

        # Compute original accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        original_acc = correct / total

        # Save original state
        state_dict = {k: v.clone() for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)}

        # Measure accuracy with perturbations
        perturbed_accs = []
        for _ in range(samples):
            # Apply random perturbation
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        noise = torch.randn_like(param) * sigma * param.norm()
                        param.add_(noise)

            # Compute accuracy with perturbation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            perturbed_acc = correct / total
            perturbed_accs.append(perturbed_acc)

            # Restore original weights
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in state_dict:
                        param.copy_(state_dict[name])

        # Calculate sharpness metrics
        avg_perturbed_acc = sum(perturbed_accs) / len(perturbed_accs)
        avg_drop = original_acc - avg_perturbed_acc
        max_drop = original_acc - min(perturbed_accs)

        sharpness_data = {
            "original_acc": original_acc,
            "avg_perturbed_acc": avg_perturbed_acc,
            "avg_drop": avg_drop,
            "max_drop": max_drop,
            "sharpness_metric": avg_drop / sigma,  # Higher means sharper minimum
            "samples": samples,
            "sigma": sigma,
            "timestamp": time.time()
        }

        # Add to sharpness metrics
        self.sharpness_metrics.append(sharpness_data)

        # Save optimization metrics
        optim_metrics = {
            "gradient_norms": self.gradient_norms,
            "weight_norms": self.weight_norms,
            "sharpness_metrics": self.sharpness_metrics
        }
        self._save_metrics("optimization_metrics.json", optim_metrics)

        return sharpness_data

    def save_final_metrics(self, best_acc, final_epoch, total_time):
        """Save final metrics at the end of training"""
        if not self.metrics_dir:
            return

        self.final_metrics = {
            "best_accuracy": best_acc,
            "final_epoch": final_epoch,
            "total_training_time": total_time,
            "epochs_to_best": self._find_epoch_to_best_acc(),
            "final_train_acc": self.train_acc_history[-1] if self.train_acc_history else None,
            "final_val_acc": self.val_acc_history[-1] if self.val_acc_history else None,
            "generalization_gap": self._calculate_generalization_gap(),
            "timestamp": time.time()
        }

        # Save final metrics
        self._save_metrics("final_metrics.json", self.final_metrics)

        # Save end timestamp
        self._save_timestamp("end")

        # Save all metrics in one file for convenience
        all_metrics = {
            "config": self.config,
            "epochs": self.epochs,
            "train_loss_history": self.train_loss_history,
            "train_acc_history": self.train_acc_history,
            "val_loss_history": self.val_loss_history,
            "val_acc_history": self.val_acc_history,
            "lr_history": self.lr_history,
            "final_metrics": self.final_metrics,
            "gradient_norms": self.gradient_norms,
            "weight_norms": self.weight_norms,
            "sharpness_metrics": self.sharpness_metrics,
            "noise_metrics": self.noise_metrics
        }

        self._save_metrics("all_metrics.json", all_metrics)

        print(f"All metrics saved to {self.metrics_dir}")

    def _find_epoch_to_best_acc(self):
        """Find the epoch where best accuracy was achieved"""
        if not self.val_acc_history:
            return None

        best_acc = max(self.val_acc_history)
        best_epoch_idx = self.val_acc_history.index(best_acc)

        return self.epochs[best_epoch_idx]

    def _calculate_generalization_gap(self):
        """Calculate generalization gap metrics"""
        if not self.train_acc_history or not self.val_acc_history:
            return {}

        # Get the gap at each epoch
        gaps = [train - val for train, val in zip(self.train_acc_history, self.val_acc_history)]

        # Calculate gap at final epoch
        final_gap = gaps[-1] if gaps else None

        # Calculate gap at best validation accuracy
        best_val_idx = self.val_acc_history.index(max(self.val_acc_history))
        best_gap = gaps[best_val_idx]

        # Calculate average gap over last 10 epochs
        if len(gaps) >= 10:
            avg_last_10_gap = sum(gaps[-10:]) / len(gaps[-10:])
        else:
            avg_last_10_gap = sum(gaps) / len(gaps)

        return {
            "final_gap": final_gap,
            "best_val_acc_gap": best_gap,
            "avg_last_10_epochs_gap": avg_last_10_gap
        }

    def _save_timestamp(self, label):
        """Save a timestamp with a label"""
        if not self.metrics_dir:
            return

        timestamp_file = os.path.join(self.metrics_dir, f"timestamp_{label}.json")
        timestamp_data = {
            "timestamp": time.time(),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "label": label
        }

        with open(timestamp_file, 'w') as f:
            json.dump(timestamp_data, f, indent=2)

    def _save_metrics(self, filename, data):
        """Save metrics to a JSON file"""
        if not self.metrics_dir:
            return

        filepath = os.path.join(self.metrics_dir, filename)
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving metrics to {filepath}: {e}")

    def _load_metrics(self, filename, default=None):
        """Load metrics from a JSON file or return default if file doesn't exist"""
        if not self.metrics_dir:
            return default

        filepath = os.path.join(self.metrics_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading metrics from {filepath}: {e}")
                return default
        return default


def evaluate_model(model, data_loader, device, criterion):
    """Evaluate model on the provided data loader"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # Initialize per-class metrics
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for images, targets in data_loader:
            # Move data to device
            images, targets = images.to(device), targets.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Update statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Per-class accuracy
            c = (predicted == targets).squeeze()
            for i in range(targets.size(0)):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # Calculate metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total

    # Calculate per-class accuracy
    class_accuracy = {}
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    for i in range(10):
        class_accuracy[classes[i]] = {
            'accuracy': class_correct[i] / class_total[i] if class_total[i] > 0 else 0,
            'correct': int(class_correct[i]),
            'total': class_total[i]
        }

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'class_accuracy': class_accuracy
    }