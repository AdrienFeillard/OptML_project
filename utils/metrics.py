import torch
import json
import os
import time
from datetime import datetime
from torch.utils.data import Subset

class TrainingMetrics:
    def __init__(self, experiment_name=None, save_dir="checkpoints/metrics", max_history=100):
        self.max_history = max_history
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.lr_history = []
        self.epochs = []

        self.current_epoch_losses = []
        self.current_epoch_accs = []
        self.current_batch_lrs = []

        self.experiment_name = experiment_name
        if experiment_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{experiment_name}_{timestamp}"
            self.metrics_dir = os.path.join(save_dir, self.experiment_name)
            os.makedirs(self.metrics_dir, exist_ok=True)
            print(f"Metrics will be saved to: {self.metrics_dir}")
        else:
            self.metrics_dir = None

        self.gradient_norms = {}
        self.weight_norms = []
        self.noise_metrics = []
        self.sharpness_metrics = []
        self.config = {}
        self.final_metrics = {}
        self.start_time = time.time()
        self.weight_update_norm_history = []
        self.optimizer_state_history = []
        self.optimization_flags_history = []
        self.avg_grad_norm_history = []
        if self.metrics_dir:
            self._save_timestamp("start")

    def set_config(self, args):
        for key, value in vars(args).items():
            if hasattr(value, 'value'):
                self.config[key] = value.value
            else:
                try:
                    json.dumps(value)
                    self.config[key] = value
                except (TypeError, OverflowError):
                    self.config[key] = str(value)

        if self.metrics_dir:
            self._save_metric_overwrite("config.json", self.config)

    def add_batch_metrics(self, loss, acc, lr):
        self.current_epoch_losses.append(loss)
        self.current_epoch_accs.append(acc)
        self.current_batch_lrs.append(lr)


    def add_epoch_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc, lr, avg_grad_norm, train_class_metrics: dict = None, val_class_metrics: dict = None):
        self.epochs.append(epoch)
        self.train_loss_history.append(train_loss)
        self.train_acc_history.append(train_acc)
        self.val_loss_history.append(val_loss)
        self.val_acc_history.append(val_acc)
        self.lr_history.append(lr)
        self.avg_grad_norm_history.append(avg_grad_norm)

        if len(self.epochs) > self.max_history:
            self.epochs.pop(0)
            self.train_loss_history.pop(0)
            self.train_acc_history.pop(0)
            self.val_loss_history.pop(0)
            self.val_acc_history.pop(0)
            self.lr_history.pop(0)
            self.avg_grad_norm_history.pop(0)

        if self.metrics_dir:
            epoch_data = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": lr,
                # Add the new dictionaries to be saved in the JSON file
                "training_per_class_metrics": train_class_metrics,
                "validation_per_class_metrics": val_class_metrics,
                "timestamp": time.time()
            }
            self._append_line_to_jsonl("epoch_metrics.jsonl", epoch_data)

    def reset_batch_metrics(self):
        self.current_epoch_losses = []
        self.current_epoch_accs = []
        self.current_batch_lrs = []

    def log_gradient_norms(self, model, epoch, batch_idx=None):
        grad_norms = {}
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                grad_norms[name] = param_norm
                total_norm += param_norm ** 2
        total_norm = total_norm ** 0.5
        norm_data = {
            "epoch": epoch, "batch": batch_idx, "total_norm": total_norm, "timestamp": time.time()
        }
        if epoch not in self.gradient_norms:
            self.gradient_norms[epoch] = []
        self.gradient_norms[epoch].append(norm_data)
        return total_norm

    def log_weight_norms(self, model, epoch, batch_idx=None):
        weight_norms = {}
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_norm = param.data.norm(2).item()
                weight_norms[name] = param_norm
                total_norm += param_norm ** 2
        total_norm = total_norm ** 0.5
        norm_data = {
            "epoch": epoch, "batch": batch_idx, "total_norm": total_norm,
            "layer_norms": weight_norms, "timestamp": time.time()
        }
        self.weight_norms.append(norm_data)
        return total_norm

    def log_noise_metrics(self, noise_type, noise_magnitude, norms_before=None, norms_after=None, epoch=None):
        if not self.metrics_dir: return
        noise_data = {"epoch": epoch, "noise_type": noise_type, "magnitude": noise_magnitude, "norms_before": norms_before, "norms_after": norms_after, "timestamp": time.time()}
        self.noise_metrics.append(noise_data)
        self._append_line_to_jsonl("noise_metrics.jsonl", noise_data)

    def log_optimization_flag(self, epoch: int, flag_type: str, details: dict = None):
        if not self.metrics_dir: return
        flag_data = {"epoch": epoch, "flag_type": flag_type, "details": details if details is not None else {}, "timestamp": time.time()}
        self.optimization_flags_history.append(flag_data)
        self._append_line_to_jsonl("optimization_flags.jsonl", flag_data)

    def measure_sharpness(self, model, criterion, val_loader, device, samples=10, sigma=0.01):
        if not self.metrics_dir: return None
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        original_acc = correct / total
        state_dict = {k: v.clone() for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)}
        perturbed_accs = []
        for _ in range(samples):
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        noise = torch.randn_like(param) * sigma * param.norm()
                        param.add_(noise)
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            perturbed_acc = correct / total
            perturbed_accs.append(perturbed_acc)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in state_dict:
                        param.copy_(state_dict[name])
        avg_perturbed_acc = sum(perturbed_accs) / len(perturbed_accs) if perturbed_accs else original_acc
        avg_drop = original_acc - avg_perturbed_acc
        sharpness_data = {"original_acc": original_acc, "avg_perturbed_acc": avg_perturbed_acc, "avg_drop": avg_drop, "sharpness_metric": avg_drop / sigma, "timestamp": time.time()}
        self.sharpness_metrics.append(sharpness_data)
        return sharpness_data

    def save_final_metrics(self, best_acc, final_epoch, total_time):
        if not self.metrics_dir: return
        self.final_metrics = {
            "best_accuracy": best_acc, "final_epoch": final_epoch, "total_training_time": total_time,
            "epochs_to_best": self._find_epoch_to_best_acc(),
            "final_train_acc": self.train_acc_history[-1] if self.train_acc_history else None,
            "final_val_acc": self.val_acc_history[-1] if self.val_acc_history else None,
            "generalization_gap": self._calculate_generalization_gap(), "timestamp": time.time()
        }
        self._save_metric_overwrite("final_metrics.json", self.final_metrics)
        self._save_timestamp("end")
        all_metrics = {
            "config": self.config, "final_metrics": self.final_metrics,
            "gradient_norms_by_epoch": self.gradient_norms, "weight_norms": self.weight_norms,
            "sharpness_metrics": self.sharpness_metrics, "noise_metrics": self.noise_metrics,
            "weight_update_norm_history": self.weight_update_norm_history,
            "optimizer_state_history": self.optimizer_state_history,
            "optimization_flags_history": self.optimization_flags_history
        }
        self._save_metric_overwrite("all_metrics.json", all_metrics)
        print(f"All metrics saved to {self.metrics_dir}")

    def _find_epoch_to_best_acc(self):
        if not self.val_acc_history: return None
        best_acc = max(self.val_acc_history)
        best_epoch_idx = self.val_acc_history.index(best_acc)
        return self.epochs[best_epoch_idx]

    def _calculate_generalization_gap(self):
        if not self.train_acc_history or not self.val_acc_history: return {}
        gaps = [train - val for train, val in zip(self.train_acc_history, self.val_acc_history)]
        final_gap = gaps[-1] if gaps else None
        best_val_idx = self.val_acc_history.index(max(self.val_acc_history))
        best_gap = gaps[best_val_idx]
        avg_last_10_gap = sum(gaps[-10:]) / len(gaps[-10:]) if len(gaps) >= 10 else sum(gaps) / len(gaps) if gaps else 0
        return {"final_gap": final_gap, "best_val_acc_gap": best_gap, "avg_last_10_epochs_gap": avg_last_10_gap}

    def log_weight_update_norm(self, model_now: torch.nn.Module, prev_weights_state: dict, epoch: int):
        if not prev_weights_state:
            self.weight_update_norm_history.append({"epoch": epoch, "norm": 0.0})
            return
        current_weights_data = {name: param.data.clone() for name, param in model_now.named_parameters() if param.requires_grad}
        total_update_norm_sq = 0.0
        for name, current_p_data in current_weights_data.items():
            if name in prev_weights_state:
                update = current_p_data - prev_weights_state[name]
                total_update_norm_sq += torch.norm(update, p=2).item() ** 2
        total_update_norm = total_update_norm_sq ** 0.5
        self.weight_update_norm_history.append({'epoch': epoch, 'norm': total_update_norm})
        if len(self.weight_update_norm_history) > self.max_history:
            self.weight_update_norm_history.pop(0)

    def log_optimizer_state(self, optimizer: torch.optim.Optimizer, epoch: int):
        opt_state_summary = {'epoch': epoch}
        if isinstance(optimizer, torch.optim.Adam):
            m_norms, v_norms, eff_lrs = [], [], []
            lr_group = optimizer.param_groups[0]['lr']
            eps = optimizer.defaults['eps']
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None and p in optimizer.state:
                        state = optimizer.state[p]
                        if 'exp_avg' in state: m_norms.append(torch.norm(state['exp_avg'], p=2).item())
                        if 'exp_avg_sq' in state:
                            v_t = state['exp_avg_sq']
                            v_norms.append(torch.norm(v_t, p=2).item())
                            eff_lrs.append((lr_group / (torch.sqrt(v_t) + eps)).mean().item())
            opt_state_summary['avg_m_norm'] = sum(m_norms) / len(m_norms) if m_norms else 0
            opt_state_summary['avg_v_norm'] = sum(v_norms) / len(v_norms) if v_norms else 0
            opt_state_summary['avg_eff_lr'] = sum(eff_lrs) / len(eff_lrs) if eff_lrs else 0
        elif isinstance(optimizer, torch.optim.SGD):
            momentum_norms = []
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None and p in optimizer.state and 'momentum_buffer' in optimizer.state[p]:
                        momentum_norms.append(torch.norm(optimizer.state[p]['momentum_buffer'], p=2).item())
            opt_state_summary['avg_momentum_norm'] = sum(momentum_norms) / len(momentum_norms) if momentum_norms else 0
        self.optimizer_state_history.append(opt_state_summary)
        if len(self.optimizer_state_history) > self.max_history:
            self.optimizer_state_history.pop(0)

    def _save_timestamp(self, label):
        if not self.metrics_dir: return
        timestamp_file = os.path.join(self.metrics_dir, f"timestamp_{label}.json")
        timestamp_data = {"timestamp": time.time(), "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "label": label}
        self._save_metric_overwrite(timestamp_file, timestamp_data)

    def _append_line_to_jsonl(self, filename, data_dict):
        if not self.metrics_dir: return
        filepath = os.path.join(self.metrics_dir, filename)
        try:
            with open(filepath, 'a') as f:
                f.write(json.dumps(data_dict) + '\n')
        except Exception as e:
            print(f"Error appending metrics to {filepath}: {e}")

    def _save_metric_overwrite(self, filename, data_dict):
        if not self.metrics_dir: return
        filepath = os.path.join(self.metrics_dir, filename)
        try:
            with open(filepath, 'w') as f:
                json.dump(data_dict, f, indent=2)
        except Exception as e:
            print(f"Error saving metrics to {filepath}: {e}")

    def get_current_epoch_avg_metrics(self):
        if not self.current_epoch_losses: return 0.0, 0.0, 0.0
        avg_loss = sum(self.current_epoch_losses) / len(self.current_epoch_losses)
        avg_acc = sum(self.current_epoch_accs) / len(self.current_epoch_accs)
        last_lr = self.current_batch_lrs[-1] if self.current_batch_lrs else 0.0
        return avg_loss, avg_acc, last_lr


# In metrics.py

def evaluate_model(model, data_loader, device, criterion):
    model.eval()

    # We still need to set reduction='none' if we want per-class loss
    # and reset it at the end.
    original_reduction = criterion.reduction
    criterion.reduction = 'none'

    total_loss_sum = 0.0
    total_correct = 0
    total_samples = 0

    num_classes = len(data_loader.dataset.dataset.classes) if isinstance(data_loader.dataset, Subset) else len(data_loader.dataset.classes)
    classes = data_loader.dataset.dataset.classes if isinstance(data_loader.dataset, Subset) else data_loader.dataset.classes

    # --- Initialize trackers on the CPU is fine here, as the loop is shorter ---
    epoch_class_correct = torch.zeros(num_classes, dtype=torch.long)
    epoch_class_loss = torch.zeros(num_classes, dtype=torch.float)
    epoch_class_total = torch.zeros(num_classes, dtype=torch.long)

    with torch.no_grad():
        for images, targets in data_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss_vector = criterion(outputs, targets) # This is a vector

            total_loss_sum += loss_vector.sum().item()

            _, predicted = outputs.max(1)
            correct_mask = (predicted == targets)

            # --- Use the fast, vectorized method ---
            epoch_class_total += torch.bincount(targets, minlength=num_classes).cpu()
            epoch_class_correct += torch.bincount(targets[correct_mask], minlength=num_classes).cpu()
            epoch_class_loss += torch.bincount(targets, weights=loss_vector, minlength=num_classes).cpu()
            # ---

    # Restore the original state of the loss function
    criterion.reduction = original_reduction

    # Calculate overall metrics
    total_samples = epoch_class_total.sum().item()
    total_correct = epoch_class_correct.sum().item()
    avg_loss = total_loss_sum / total_samples if total_samples > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0

    # --- Final Dictionary Creation (with the range(10) bug fixed) ---
    class_accuracy = {}
    for i in range(num_classes): # Use dynamic num_classes
        total_i = epoch_class_total[i].item()
        class_accuracy[classes[i]] = {
            'accuracy': epoch_class_correct[i].item() / total_i if total_i > 0 else 0,
            'avg_loss': epoch_class_loss[i].item() / total_i if total_i > 0 else 0,
            'correct': epoch_class_correct[i].item(),
            'total': total_i
        }

    return {'loss': avg_loss, 'accuracy': accuracy, 'correct': total_correct, 'total': total_samples, 'class_accuracy': class_accuracy}

