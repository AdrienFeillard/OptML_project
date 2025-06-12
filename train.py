import os
import time
import datetime
import typer
import torch
#import torch_directml
import numpy as np
from typing import Optional, List, Tuple, Any, Dict

from rich.console import Console, Group
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.traceback import install as install_rich_traceback
from torch.utils.data import Subset

from core.data import CIFARData
from core.module import CIFAR10Module, Optimizer as OptimizerChoice
from utils.logging import LogHandler
from utils.metrics import TrainingMetrics, evaluate_model
from utils.visualization import create_dashboard, visualize_lr_schedule
from utils.configs.config import Classifier, NoiseDistribution
from core.noise_regularization import NoiseType, NoiseSchedule
from utils.visualization import (
    create_dashboard,
    update_dashboard_contents,
    visualize_lr_schedule
)

install_rich_traceback(show_locals=True)
console = Console()


class ArgsNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _create_args_namespace(
        data_dir: str, checkpoint_name: str, download_weights: bool, classifier: Classifier,
        batch_size: int, max_epochs: int, num_workers: int, gpu_id: str,
        learning_rate: float, weight_decay: float,
        noise_type: NoiseType, noise_magnitude: float, noise_schedule: NoiseSchedule,
        noise_layer: Optional[List[str]], noise_distribution: NoiseDistribution,
        optimizer: OptimizerChoice, momentum: float, beta1: float, beta2: float,
        permanent: bool, noise_during_stuck_only: bool, patience: int, subset: float,
        visualize_lr: bool, flag_min_epochs_to_check: int, flag_window_size: int,
        flag_overfitting_val_loss_epochs: float, flag_plateau_min_delta: float,
        flag_grad_plateau_thr: float, flag_low_weight_update_threshold: float,
        disable_adaptive_flags: bool,
        disable_graphs: bool,
        relative_min_noise: float,
        relative_max_noise: float,
        consecutive_flag_trigger: int,
        min_cooldown_epochs: int,
        max_cooldown_epochs: int,
        lr_restart_period: int,
        detailed_metrics: bool,
) -> ArgsNamespace:
    return ArgsNamespace(
        data_dir=data_dir, checkpoint_name=checkpoint_name, download_weights=download_weights,
        classifier=classifier, batch_size=batch_size, max_epochs=max_epochs, num_workers=num_workers,
        gpu_id=gpu_id, learning_rate=learning_rate, weight_decay=weight_decay,
        noise_type=noise_type, noise_magnitude=noise_magnitude, noise_schedule=noise_schedule,
        noise_layer=noise_layer, noise_distribution=noise_distribution, optimizer=optimizer,
        momentum=momentum, beta1=beta1, beta2=beta2, permanent=permanent,
        noise_during_stuck_only=noise_during_stuck_only, patience=patience, subset=subset,
        test_phase=False, visualize_lr=visualize_lr, model=None,
        flag_min_epochs_to_check=flag_min_epochs_to_check, flag_window_size=flag_window_size,
        flag_overfitting_val_loss_epochs=flag_overfitting_val_loss_epochs,
        flag_plateau_min_delta=flag_plateau_min_delta,
        flag_grad_plateau_thr=flag_grad_plateau_thr,
        flag_low_weight_update_threshold=flag_low_weight_update_threshold,
        disable_adaptive_flags=disable_adaptive_flags,
        disable_graphs=disable_graphs,
        relative_min_noise=relative_min_noise,
        relative_max_noise=relative_max_noise,
        consecutive_flag_trigger=consecutive_flag_trigger,
        min_cooldown_epochs=min_cooldown_epochs,
        max_cooldown_epochs=max_cooldown_epochs,
        lr_restart_period=lr_restart_period,
        detailed_metrics=detailed_metrics,
    )


def _setup_environment(args: ArgsNamespace, log_handler: LogHandler) -> torch.device:
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    log_handler.log("SYSTEM", "Random seed set to 0 for reproducibility")
    try:
        device_id = int(args.gpu_id.split(",")[0])
        log_handler.log("SYSTEM", f"Initializing DirectML device {device_id}")
        device = torch_directml.device(device_id)
        log_handler.log("SYSTEM", f"Using DirectML device: {device}")
    except Exception as e:
        log_handler.log("ERROR", f"Error initializing DirectML: {e}")
        log_handler.log("SYSTEM", "Falling back to CPU")

        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("CUDA USED")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
            print("MPS USED")
        else:
            device = torch.device("cpu")
            print("CPU USED")

    return device


def _initialize_training_components(args: ArgsNamespace) -> Tuple[TrainingMetrics, LogHandler]:
    experiment_name = f"{args.classifier}_{args.noise_type}"
    if args.noise_type != NoiseType.none:
        experiment_name += f"_{args.noise_distribution}_{args.noise_magnitude}"
    training_metrics = TrainingMetrics(experiment_name=experiment_name)
    training_metrics.set_config(args)
    log_handler = LogHandler()
    log_handler.log("SYSTEM", f"Starting CIFAR-10 training with {args.classifier}")
    return training_metrics, log_handler


def _prepare_data_and_model(args: ArgsNamespace, device: torch.device, log_handler: LogHandler) -> Tuple[
    CIFAR10Module, CIFARData]:
    log_handler.log("MODEL", f"Creating {args.classifier} model...")
    # The model now depends on args.num_classes, which must be set first.
    model = CIFAR10Module(args)
    log_handler.log("DATA", f"Preparing {args.dataset_name} dataset...")
    # The data module is now created outside this function, so we just receive it.
    cifar_data = args.cifar_data_module
    model.to(device)
    log_handler.log("MODEL", f"Model moved to device: {device}")
    return model, cifar_data


def _configure_optimizer_and_scheduler(model: CIFAR10Module, args: ArgsNamespace, train_loader_len: int,
                                       log_handler: LogHandler) -> Tuple[torch.optim.Optimizer, Any]:
    # Pass the train_loader_len (steps_per_epoch) to the model's method
    optimizer, scheduler = model.configure_optimizer(steps_per_epoch=train_loader_len)
    log_handler.log("OPTIMIZER",
                    f"Created optimizer: {args.optimizer.value}, lr={args.learning_rate}, wd={args.weight_decay}")
    if args.visualize_lr:
        total_steps = args.max_epochs * train_loader_len
        vis_optimizer, vis_scheduler = model.configure_optimizer(steps_per_epoch=train_loader_len)
        log_handler.log("LR_SCHEDULE", "Visualizing learning rate schedule...")
        visualize_lr_schedule(vis_optimizer, vis_scheduler, total_steps)
    return optimizer, scheduler


def _apply_noise_regularization_epoch_start(model: CIFAR10Module, epoch: int, training_metrics: TrainingMetrics,
                                            log_handler: LogHandler):
    """
    Logs the noise regularizer's current state without recalculating it.
    """
    if hasattr(model, 'noise_regularizer') and model.noise_regularizer:
        # --- THIS IS THE FIX ---
        # DO NOT call model.noise_regularizer.update_epoch(epoch) here.
        # The magnitude is now calculated adaptively inside _train_epoch.
        # We just need to read the value that was already set.
        current_magnitude = model.noise_regularizer.current_magnitude
        # --- END OF FIX ---

        training_metrics.log_noise_metrics(
            noise_type=str(model.noise_regularizer.noise_type.value),
            noise_magnitude=current_magnitude,
            epoch=epoch + 1
        )
        # Only log if noise is actually ON
        if model.noise_regularizer.noise_type != NoiseType.none:
            log_handler.log("NOISE",
                            f"Applying {model.noise_regularizer.noise_type.value} noise with magnitude {current_magnitude:.6f}")



# Renamed and modified to specifically detect flags and return their state
def _detect_stagnation_flags(
        args: ArgsNamespace, training_metrics: TrainingMetrics, log_handler: LogHandler, current_epoch: int
) -> tuple[dict[str, bool], dict[str, float]]:
    flag_states = {"grad_norm_plateau": False, "low_weight_update": False, "val_loss_plateau": False,
                   "overfitting": False}
    flag_details = {}

    if current_epoch < args.flag_min_epochs_to_check:
        return flag_states, flag_details

    WINDOW_SIZE = args.flag_window_size
    PLATEAU_DELTA = args.flag_plateau_min_delta

    train_loss_history = training_metrics.train_loss_history
    val_loss_history = training_metrics.val_loss_history
    weight_update_norm_history = training_metrics.weight_update_norm_history
    #gradient_norms = training_metrics.gradient_norms

    # Low Gradient Norm Flag
    avg_grad_norm_history = training_metrics.avg_grad_norm_history
    if len(avg_grad_norm_history) >= WINDOW_SIZE:
        recent_norms = avg_grad_norm_history[-WINDOW_SIZE:]

        # Check for non-zero to avoid division errors
        if recent_norms[0] > 1e-9:
            # A positive value means the norm has decreased (which is good)
            relative_improvement = (recent_norms[0] - recent_norms[-1]) / recent_norms[0]

            # If improvement is less than our threshold, it has plateaued
            if relative_improvement < args.flag_grad_plateau_thr:
                log_handler.log("FLAG_DETECTED",
                                f"Epoch {current_epoch}: GRADIENT NORM PLATEAU (improvement: {relative_improvement:.2%}).")
                training_metrics.log_optimization_flag(current_epoch, "GRAD_NORM_PLATEAU",
                                                       {"relative_improvement": relative_improvement})
                flag_states["grad_norm_plateau"] = True
                flag_details["grad_norm_relative_improvement"] = relative_improvement

    # Low Weight Update Norm Flag
    if len(weight_update_norm_history) >= current_epoch and current_epoch > 0:
        relevant_history = [h['norm'] for h in weight_update_norm_history if h['epoch'] <= current_epoch][-WINDOW_SIZE:]
        if len(relevant_history) == WINDOW_SIZE:
            avg_recent_update_norm = sum(relevant_history) / WINDOW_SIZE
            if avg_recent_update_norm < args.flag_low_weight_update_threshold:
                log_handler.log("FLAG_DETECTED",
                                f"Epoch {current_epoch}: LOW WEIGHT UPDATE NORM (avg: {avg_recent_update_norm:.2e}).")
                training_metrics.log_optimization_flag(current_epoch, "LOW_WEIGHT_UPDATE_NORM",
                                                       {"avg_norm": avg_recent_update_norm})
                flag_states["low_weight_update"] = True

    # Validation Loss Plateau Flag
    if len(val_loss_history) >= current_epoch and current_epoch > 0:
        recent_val_losses = val_loss_history[max(0, current_epoch - WINDOW_SIZE): current_epoch]
        if len(recent_val_losses) == WINDOW_SIZE:
            net_improvement = recent_val_losses[0] - recent_val_losses[-1]
            if net_improvement < PLATEAU_DELTA:
                log_handler.log("FLAG_DETECTED",
                                f"Epoch {current_epoch}: Validation Loss PLATEAU (net improvement {net_improvement:.2e} < {PLATEAU_DELTA:.2e}).")
                training_metrics.log_optimization_flag(current_epoch, "VAL_LOSS_PLATEAU",
                                                       {"net_improvement": net_improvement})
                flag_states["val_loss_plateau"] = True
                flag_details["val_loss_net_improvement"] = net_improvement

    if len(val_loss_history) >= current_epoch and len(train_loss_history) >= current_epoch and current_epoch > 0:
        recent_val_losses = val_loss_history[max(0, current_epoch - WINDOW_SIZE): current_epoch]
        recent_train_losses = train_loss_history[max(0, current_epoch - WINDOW_SIZE): current_epoch]

        if len(recent_val_losses) == WINDOW_SIZE and len(recent_train_losses) == WINDOW_SIZE:
            val_loss_increasing_count = sum(1 for i in range(1, WINDOW_SIZE) if
                                            recent_val_losses[i] > recent_val_losses[i - 1] + PLATEAU_DELTA / 10)
            train_loss_decreasing_count = sum(1 for i in range(1, WINDOW_SIZE) if
                                              recent_train_losses[i] < recent_train_losses[i - 1] - PLATEAU_DELTA / 10)
            if val_loss_increasing_count >= args.flag_overfitting_val_loss_epochs and \
                    train_loss_decreasing_count >= args.flag_overfitting_val_loss_epochs:
                log_handler.log("FLAG_DETECTED", f"Epoch {current_epoch}: Potential OVERFITTING. Val loss trend up, "
                                                 f"Train loss trend down.")
                training_metrics.log_optimization_flag(current_epoch, "OVERFITTING",
                                                       {"val_loss_increasing_epochs": val_loss_increasing_count,
                                                        "train_loss_decreasing_epochs": train_loss_decreasing_count})
                flag_states["overfitting"] = True

    return flag_states, flag_details


def _apply_weight_noise_step(
        model: CIFAR10Module, args: ArgsNamespace, permanent: bool,
        epoch: int, batch_idx: int, training_metrics: TrainingMetrics, log_handler: LogHandler
) -> bool:
    original_weights_saved_this_batch = False
    if hasattr(model, 'noise_regularizer') and model.noise_regularizer and \
            model.noise_regularizer.noise_type == NoiseType.weight:
        if not permanent:
            model.noise_regularizer.save_original_weights(model)
            original_weights_saved_this_batch = True
        # Log this specific application
        #log_handler.log("NOISE_APPLY", f"Applying weight noise (magnitude: {model.noise_regularizer.current_magnitude:.6f}, permanent: {permanent}).")
        model.noise_regularizer.apply_weight_noise(model, permanent=permanent)
        # Log metrics before/after noise apply for analysis
        weight_norms_before = training_metrics.log_weight_norms(model, epoch + 1, batch_idx)
        weight_norms_after = training_metrics.log_weight_norms(model, epoch + 1, batch_idx)
        if batch_idx % 50 == 0:
            training_metrics.log_noise_metrics(
                noise_type=str(NoiseType.weight.value), noise_magnitude=model.noise_regularizer.current_magnitude,
                norms_before=weight_norms_before, norms_after=weight_norms_after, epoch=epoch + 1
            )
    return original_weights_saved_this_batch


def _apply_label_noise_step(model: CIFAR10Module, targets: torch.Tensor) -> torch.Tensor:
    if hasattr(model, 'noise_regularizer') and model.noise_regularizer and \
            model.noise_regularizer.noise_type == NoiseType.label:
        return model.noise_regularizer.apply_label_noise(targets)
    return targets


def _restore_weights_step(model: CIFAR10Module, original_weights_saved_this_batch: bool):
    if original_weights_saved_this_batch:
        model.noise_regularizer.restore_weights(model)


def _apply_gradient_noise_step(
        model: CIFAR10Module, epoch: int, batch_idx: int, training_metrics: TrainingMetrics, log_handler: LogHandler
):
    # Log before even if not applying noise, to get baseline for the epoch
    grad_norms_before = training_metrics.log_gradient_norms(model, epoch + 1, batch_idx)

    if hasattr(model, 'noise_regularizer') and model.noise_regularizer and \
            model.noise_regularizer.noise_type == NoiseType.gradient:
        # Log this specific application
        #log_handler.log("NOISE_APPLY", f"Applying gradient noise (magnitude: {model.noise_regularizer.current_magnitude:.6f}).")
        model.noise_regularizer.apply_gradient_noise(model)
        # Log after noise apply for analysis
        grad_norms_after = training_metrics.log_gradient_norms(model, epoch + 1, batch_idx)

        if batch_idx % 50 == 0:
            training_metrics.log_noise_metrics(
                noise_type=str(NoiseType.gradient.value), noise_magnitude=model.noise_regularizer.current_magnitude,
                norms_before=grad_norms_before, norms_after=grad_norms_after, epoch=epoch + 1
            )


# Modified to handle noise activation/deactivation dynamically
def _train_epoch(
        epoch: int, model: CIFAR10Module, train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer, scheduler: Any, device: torch.device,
        args: ArgsNamespace, training_metrics: TrainingMetrics, log_handler: LogHandler,
        apply_grad_noise_this_epoch: bool, apply_weight_noise_this_epoch: bool,
        noise_active_since_epoch: List[Optional[int]],
        batch_progress_widget: Progress,
        batch_task_id: Any,
        dashboard_components: Dict[str, Any],
        max_epochs_for_dashboard: int,
        is_stuck_for_patience: bool,
        flag_details: dict
) -> Tuple[float, int, int, dict]:
    model.train()
    train_correct, train_total = 0, 0
    epoch_total_loss = 0.0

    # Conditionally set up detailed trackers
    if args.detailed_metrics:
        num_classes = args.num_classes
        class_names = train_loader.dataset.dataset.classes if isinstance(train_loader.dataset, Subset) else train_loader.dataset.classes
        epoch_class_correct = torch.zeros(num_classes, dtype=torch.long, device=device)
        epoch_class_loss = torch.zeros(num_classes, dtype=torch.float, device=device)
        epoch_class_total = torch.zeros(num_classes, dtype=torch.long, device=device)
        model.criterion.reduction = 'none'
    else:
        model.criterion.reduction = 'mean'

    # --- Start of Noise Logic ---
    desired_noise_type_for_this_epoch = NoiseType.none
    if args.noise_during_stuck_only:  # User's Requirement: If True, ONLY adaptive noise
        # Noise is currently inactive, check if any adaptive flag should activate it
        if apply_grad_noise_this_epoch:
            desired_noise_type_for_this_epoch = NoiseType.gradient
            log_handler.log("NOISE_ACTIVATED",
                            f"Epoch {epoch + 1}: Activating GRADIENT noise due to adaptive flags (Low Grad Norm / Low Weight Update).")
            noise_active_since_epoch[0] = epoch + 1  # Mark epoch when noise became active
        elif apply_weight_noise_this_epoch:
            desired_noise_type_for_this_epoch = NoiseType.weight
            log_handler.log("NOISE_ACTIVATED",
                            f"Epoch {epoch + 1}: Activating WEIGHT noise due to adaptive flags (Val Loss Plateau / Overfitting).")
            noise_active_since_epoch[0] = epoch + 1  # Mark epoch when noise became active
        else:  # If no adaptive flags triggered and in stuck-only mode, noise remains none
            # Strategy for deactivation:
            if noise_active_since_epoch[0] is not None:
                desired_noise_type_for_this_epoch = NoiseType.none
                log_handler.log("NOISE_DEACTIVATED",
                                f"Epoch {epoch + 1}: Deactivating noise as triggering conditions are no longer met (in stuck-only mode).")
                noise_active_since_epoch[0] = None  # Mark noise as inactive
            # Else (noise was already inactive and no flags fired), desired_noise_type remains none.

    else:  # args.noise_during_stuck_only = False (User's Requirement: Continuous OR specific adaptive after patience)
        # Check if adaptive flags are triggered first (they take precedence if noise is currently inactive)
        if noise_active_since_epoch[0] is None:  # Only activate adaptively if currently inactive
            if apply_grad_noise_this_epoch:
                desired_noise_type_for_this_epoch = NoiseType.gradient
                log_handler.log("NOISE_ACTIVATED",
                                f"Epoch {epoch + 1}: Activating ADAPTIVE GRADIENT noise (not stuck-only mode).")
                noise_active_since_epoch[0] = epoch + 1
            elif apply_weight_noise_this_epoch:
                desired_noise_type_for_this_epoch = NoiseType.weight
                log_handler.log("NOISE_ACTIVATED",
                                f"Epoch {epoch + 1}: Activating ADAPTIVE WEIGHT noise (not stuck-only mode).")
                noise_active_since_epoch[0] = epoch + 1
            elif is_stuck_for_patience and args.noise_type != NoiseType.none:  # New condition for continuous after patience
                # This branch handles continuous noise *only after* patience is exceeded AND args.noise_type is set
                desired_noise_type_for_this_epoch = args.noise_type
                log_handler.log("NOISE_ACTIVATED",
                                f"Epoch {epoch + 1}: Activating CONTINUOUS noise ({args.noise_type.value}) due to patience exceeded (not stuck-only).")
                noise_active_since_epoch[0] = epoch + 1  # Mark epoch when noise became active
            else:
                # If no adaptive flags fired, patience not exceeded, and not in stuck-only mode,
                # then desired_noise_type_for_this_epoch remains NoiseType.none (baseline).
                pass  # Already initialized to NoiseType.none
        else:  # If noise was active (either adaptive or continuous from previous epoch), check conditions
            if not apply_grad_noise_this_epoch and not apply_weight_noise_this_epoch and not is_stuck_for_patience:
                # Deactivate if no adaptive flags persist AND patience condition is no longer met
                desired_noise_type_for_this_epoch = NoiseType.none
                log_handler.log("NOISE_DEACTIVATED",
                                f"Epoch {epoch + 1}: Deactivating noise as triggering conditions are no longer met (not stuck-only mode).")
                noise_active_since_epoch[0] = None
            else:
                # Keep noise active if adaptive flags persist OR patience condition is still met
                if apply_grad_noise_this_epoch:
                    desired_noise_type_for_this_epoch = NoiseType.gradient
                elif apply_weight_noise_this_epoch:
                    desired_noise_type_for_this_epoch = NoiseType.weight
                elif is_stuck_for_patience and args.noise_type != NoiseType.none:  # Keep continuous if still stuck and configured
                    desired_noise_type_for_this_epoch = args.noise_type
                log_handler.log("NOISE_STATUS",
                                f"Epoch {epoch + 1}: Noise remains active ({desired_noise_type_for_this_epoch.value}).")

    if desired_noise_type_for_this_epoch != NoiseType.none:
        stuckness_factor = 0.5  # Default stuckness
        if desired_noise_type_for_this_epoch == NoiseType.gradient and "grad_norm_relative_improvement" in flag_details:
            improvement = flag_details["grad_norm_relative_improvement"]
            threshold = args.flag_grad_plateau_thr
            k = 2 / threshold

            sigmoid_val = 1 / (1 + np.exp(-k * (improvement - threshold)))
            stuckness_factor = 1.0 - sigmoid_val
        if desired_noise_type_for_this_epoch == NoiseType.weight and "val_loss_net_improvement" in flag_details:
            improvement = flag_details["val_loss_net_improvement"]
            threshold = args.flag_plateau_min_delta
            k = 2 / threshold

            sigmoid_val = 1 / (1 + np.exp(-k * (improvement - threshold)))
            stuckness_factor = 1.0 - sigmoid_val

        raw_kick_magnitude = args.relative_min_noise + stuckness_factor * (
                    args.relative_max_noise - args.relative_min_noise)

        # 2. Calculate the "Schedule Dampening Factor"
        # This requires getting the current value of the main LR schedule
        # For simplicity, we'll simulate it here. A better way would be to query the scheduler.
        progress = epoch / args.max_epochs
        schedule_dampening_factor = 0.5 * (1 + np.cos(np.pi * progress))  # Assuming cosine

        # 3. Calculate Final Magnitude
        final_magnitude = raw_kick_magnitude * schedule_dampening_factor
        print("final magnitude",final_magnitude)
        # 4. Set it in the regularizer
        model.noise_regularizer.current_magnitude = final_magnitude
        log_handler.log("NOISE_DYNAMIC",
                        f"Dynamic noise magnitude set to {final_magnitude:.6f} (raw_kick: {raw_kick_magnitude:.4f}, schedule_dampening: {schedule_dampening_factor:.2f})")

    model.noise_regularizer.noise_type = desired_noise_type_for_this_epoch
    _apply_noise_regularization_epoch_start(model, epoch, training_metrics, log_handler)

    for batch_idx, (images, targets) in enumerate(train_loader):
        batch_start_time = time.time()
        images, targets = images.to(device), targets.to(device)

        original_weights_saved = False

        if model.noise_regularizer.noise_type == NoiseType.weight:
            original_weights_saved = _apply_weight_noise_step(model, args, args.permanent, epoch, batch_idx,
                                                              training_metrics, log_handler)

        outputs = model(images)
        loss = model.criterion(outputs, targets)

        optimizer.zero_grad()
        loss.mean().backward()

        training_metrics.log_gradient_norms(model, epoch + 1, batch_idx)

        if model.noise_regularizer.noise_type == NoiseType.weight:
            _restore_weights_step(model, original_weights_saved)
        elif model.noise_regularizer.noise_type == NoiseType.gradient:
            _apply_gradient_noise_step(model, epoch, batch_idx, training_metrics, log_handler)


        optimizer.step()
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Get overall batch accuracy once, it's fast
        total_correct_in_batch, total_in_batch = model.calculate_accuracy(outputs, targets)
        train_correct += total_correct_in_batch
        train_total += total_in_batch

        # --- CORRECTED CONDITIONAL METRIC COLLECTION ---
        if args.detailed_metrics:
            # Accumulate loss from the per-sample loss vector
            epoch_total_loss += loss.sum().item()

            correct_mask = (outputs.max(1)[1] == targets)

            for i in range(num_classes):
                # Create a mask for the current class 'i'
                class_mask = (targets == i)

                # If there are any samples of this class in the batch
                if class_mask.any():
                    # 1. Calculate total samples for class 'i'
                    epoch_class_total[i] += class_mask.sum()

                    # 2. Calculate correctly predicted samples for class 'i'
                    # We check where the target was class 'i' AND the prediction was correct
                    epoch_class_correct[i] += (class_mask & correct_mask).sum()

                    # 3. Calculate the sum of losses for class 'i'
                    # We use the class_mask to select losses only for samples of class 'i'
                    epoch_class_loss[i] += loss.detach()[class_mask].sum()

            batch_loss_for_display = loss.mean().item()
            training_metrics.add_batch_metrics(batch_loss_for_display, total_correct_in_batch / total_in_batch, current_lr)
        else:
            # Fast path: `loss` is already a scalar
            epoch_total_loss += loss.item() * total_in_batch

            batch_loss_for_display = loss.item()
            training_metrics.add_batch_metrics(batch_loss_for_display, total_correct_in_batch / total_in_batch, current_lr)


        batch_progress_widget.update(batch_task_id, advance=1)
        if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
            update_dashboard_contents(
                dashboard_components, training_metrics, log_handler, args, model,
                current_epoch=epoch + 1, max_epochs=max_epochs_for_dashboard,
                batch_idx=batch_idx + 1, len_train_loader=len(train_loader)
            )
        if batch_idx % 50 == 0:
            # Use the already calculated batch accuracy
            batch_acc = total_correct_in_batch / total_in_batch
            # Use the correct loss scalar
            batch_loss = loss.mean().item() if args.detailed_metrics else loss.item()
            log_handler.log(
                "BATCH",
                f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, "
                f"Loss: {batch_loss:.4f}, Acc: {100. * batch_acc:.2f}%, "
                f"LR: {current_lr:.6f}, Time: {time.time() - batch_start_time:.2f}s"
            )

    per_class_train_metrics = None
    if args.detailed_metrics:
        model.criterion.reduction = 'mean'  # Reset state
        # Move final totals to CPU once at the end of the epoch
        epoch_class_total = epoch_class_total.cpu()
        epoch_class_correct = epoch_class_correct.cpu()
        epoch_class_loss = epoch_class_loss.cpu()

        per_class_train_metrics = {}
        for i in range(num_classes):
            total_i = epoch_class_total[i].item()
            per_class_train_metrics[class_names[i]] = {
                'accuracy': epoch_class_correct[i].item() / total_i if total_i > 0 else 0,
                'avg_loss': epoch_class_loss[i].item() / total_i if total_i > 0 else 0,
            }

    # If not in detailed mode, use the simple counters
    if not args.detailed_metrics:
        epoch_total_loss = epoch_total_loss # Already calculated

    # Unify the return values
    final_train_correct = train_correct
    final_train_total = train_total

    return epoch_total_loss, final_train_correct, final_train_total, per_class_train_metrics


def _validate_epoch(
        epoch: int, model: CIFAR10Module, val_loader: torch.utils.data.DataLoader,
        device: torch.device, log_handler: LogHandler
) -> Tuple[float, int, int]:
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    log_handler.log("VALID", f"Starting validation for epoch {epoch + 1}")
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = model.criterion(outputs, targets)
            val_loss += loss.item()
            correct, total = model.calculate_accuracy(outputs, targets)
            val_correct += correct
            val_total += total
    return val_loss, val_correct, val_total


def _update_stuck_state(
        args: ArgsNamespace, current_epoch_train_loss: float,
        best_loss_for_stuck: float, patience_count: int
) -> Tuple[bool, float, int]:
    is_stuck_for_noise_application = False
    if args.noise_during_stuck_only:
        if current_epoch_train_loss < best_loss_for_stuck:
            best_loss_for_stuck = current_epoch_train_loss
            patience_count = 0
            is_stuck_for_noise_application = False
            console.print(f"[INFO] Training loss improved to {best_loss_for_stuck:.4f}. Patience counter reset.")
        else:
            patience_count += 1
            console.print(f"[INFO] Training loss did not improve. Patience: {patience_count}/{args.patience}")
            if patience_count > args.patience:
                is_stuck_for_noise_application = True
                console.print(f"[WARNING] Patience exceeded. Training may be stuck.")
    return is_stuck_for_noise_application, best_loss_for_stuck, patience_count


# In train.py, replace your existing function with this one.

def _run_training_loop(
        args: ArgsNamespace, model: CIFAR10Module, cifar_data: CIFARData, device: torch.device,
        optimizer: torch.optim.Optimizer, scheduler: Any,
        training_metrics: TrainingMetrics, log_handler: LogHandler
) -> Tuple[float, float, Optional[dict], float]:
    train_loader = cifar_data.train_dataloader()
    val_loader = cifar_data.val_dataloader()
    log_handler.log("DATA",
                    f"Dataset ready with {len(train_loader.dataset)} training samples and {len(val_loader.dataset)} validation samples")

    os.makedirs("checkpoints", exist_ok=True)
    log_handler.log("SYSTEM", "Created checkpoints directory")

    training_metrics.log_weight_norms(model, epoch=0)
    training_metrics.log_optimizer_state(optimizer, epoch=0)

    epoch_progress_widget = Progress(
        TextColumn("[bold blue]Epoch Progress"), BarColumn(bar_width=None),
        TaskProgressColumn(), TextColumn("{task.percentage:>3.0f}%"), TimeRemainingColumn(), expand=True,
    )
    epoch_task_id = epoch_progress_widget.add_task("Epochs", total=args.max_epochs, completed=0, start=True)

    batch_progress_widget = Progress(
        TextColumn("[bold cyan]Batch Progress"), BarColumn(bar_width=None),
        TaskProgressColumn(), TextColumn("{task.percentage:>3.0f}%"), TimeRemainingColumn(), expand=True, transient=True
    )

    prev_model_state_for_updates = {name: param.data.clone() for name, param in model.named_parameters() if
                                    param.requires_grad}

    layout, dashboard_components = create_dashboard(
        args, training_metrics, log_handler, model, epoch_progress_widget, batch_progress_widget
    )

    best_acc = 0.0
    start_time = time.time()
    is_stuck_for_patience = False
    patience_counter = 0
    best_loss_for_stuck_detection = float('inf')

    best_model_state_dict_to_save = None
    actual_best_val_acc_for_saving = 0.0
    noise_active_since_epoch = [None]

    consecutive_noise_epochs = 0
    cooldown_counter = 0

    with Live(layout, refresh_per_second=0.1, console=console, transient=False) as live:
        for epoch in range(args.max_epochs):

            epoch_start_time = time.time()
            log_handler.log("TRAIN", f"Starting epoch {epoch + 1}/{args.max_epochs}")

            apply_grad_noise_this_epoch = False
            apply_weight_noise_this_epoch = False
            flag_details = {}

            if cooldown_counter > 0:
                log_handler.log("COOLDOWN",
                                f"System in cooldown. Skipping flag checks. Epochs remaining: {cooldown_counter}")
                cooldown_counter -= 1
            elif not args.disable_adaptive_flags and epoch > 0:
                flag_states, flag_details = _detect_stagnation_flags(args, training_metrics, log_handler, epoch)
                apply_grad_noise_this_epoch = flag_states["grad_norm_plateau"] or flag_states["low_weight_update"]
                apply_weight_noise_this_epoch = flag_states["val_loss_plateau"] or flag_states["overfitting"]

            is_noise_applied_this_epoch = apply_grad_noise_this_epoch or apply_weight_noise_this_epoch

            if is_noise_applied_this_epoch:
                consecutive_noise_epochs += 1
            else:
                consecutive_noise_epochs = 0

            if consecutive_noise_epochs >= args.consecutive_flag_trigger:
                log_handler.log("COOLDOWN",
                                f"Noise triggered for {consecutive_noise_epochs} consecutive epochs. Initiating cooldown.")
                stuckness_factor = 0.5
                if "grad_norm_relative_improvement" in flag_details:
                    improvement = flag_details["grad_norm_relative_improvement"]
                    threshold = args.flag_grad_plateau_thr
                    k = 2 / threshold

                    sigmoid_val = 1 / (1 + np.exp(-k * (improvement - threshold)))
                    stuckness_factor = 1.0 - sigmoid_val
                    print('GRadient noise stuckness factor:', stuckness_factor)
                cooldown_duration = int(
                    args.min_cooldown_epochs + stuckness_factor * (args.max_cooldown_epochs - args.min_cooldown_epochs))
                cooldown_counter = cooldown_duration
                print("cooldown duration second part",
                      stuckness_factor * (args.max_cooldown_epochs - args.min_cooldown_epochs))
                log_handler.log("COOLDOWN", f"Adaptive cooldown period set to {cooldown_duration} epochs.")
                consecutive_noise_epochs = 0
            # --- END OF CONSOLIDATED LOGIC ---

            if batch_progress_widget.tasks:
                for task in list(batch_progress_widget.tasks):
                    batch_progress_widget.stop_task(task.id)
                    batch_progress_widget.remove_task(task.id)
            current_batch_task_id = batch_progress_widget.add_task(
                "Batches", total=len(train_loader), completed=0, start=True
            )

            # This separate patience logic for continuous noise mode is fine
            current_train_loss = training_metrics.train_loss_history[
                -1] if training_metrics.train_loss_history else float('inf')
            is_stuck_for_patience, best_loss_for_stuck_detection, patience_counter = _update_stuck_state(
                args, current_train_loss, best_loss_for_stuck_detection, patience_counter
            )

            epoch_train_loss_sum, epoch_train_correct, epoch_train_total, per_class_train_metrics = _train_epoch(
                epoch, model, train_loader, optimizer, scheduler, device, args,
                training_metrics, log_handler,
                apply_grad_noise_this_epoch, apply_weight_noise_this_epoch,
                noise_active_since_epoch,
                batch_progress_widget, current_batch_task_id,
                dashboard_components, args.max_epochs,
                is_stuck_for_patience,
                flag_details
            )
            epoch_progress_widget.update(epoch_task_id, advance=1)

            avg_epoch_train_loss = epoch_train_loss_sum / epoch_train_total if epoch_train_total > 0 else 0
            epoch_train_accuracy = 100. * epoch_train_correct / epoch_train_total if epoch_train_total > 0 else 0

            # --- CHANGE 2: Replace the simple _validate_epoch call with the detailed evaluate_model ---
            detailed_val_results = evaluate_model(model, val_loader, device, model.criterion)
            avg_epoch_val_loss = detailed_val_results['loss']
            epoch_val_accuracy = 100. * detailed_val_results['accuracy']
            per_class_val_metrics = detailed_val_results['class_accuracy']
            # --- End of Change 2 ---

            grad_norms_for_epoch = [item['total_norm'] for item in training_metrics.gradient_norms.get(epoch + 1, [])]
            avg_epoch_grad_norm = sum(grad_norms_for_epoch) / len(grad_norms_for_epoch) if grad_norms_for_epoch else 0

            current_lr = scheduler.get_last_lr()[0]

            # --- CHANGE 3: Pass the new detailed metrics to the logger ---
            training_metrics.add_epoch_metrics(
                epoch=epoch + 1, train_loss=avg_epoch_train_loss, train_acc=epoch_train_accuracy / 100.0,
                val_loss=avg_epoch_val_loss, val_acc=epoch_val_accuracy / 100.0, lr=current_lr,
                avg_grad_norm=avg_epoch_grad_norm,
                train_class_metrics=per_class_train_metrics,
                val_class_metrics=per_class_val_metrics
            )

            current_model_state_for_updates = {name: param.data.clone() for name, param in model.named_parameters() if
                                               param.requires_grad}
            training_metrics.log_weight_update_norm(model, prev_model_state_for_updates, epoch + 1)
            prev_model_state_for_updates = {name: param.data.clone() for name, param in model.named_parameters() if
                                            param.requires_grad}

            training_metrics.log_optimizer_state(optimizer, epoch + 1)
            log_handler.log(
                "EPOCH",
                f"Epoch {epoch + 1} completed in {time.time() - epoch_start_time:.2f}s - "
                f"Train Loss: {avg_epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.2f}%, "
                f"Val Loss: {avg_epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.2f}%"
            )

            if epoch_val_accuracy > actual_best_val_acc_for_saving:
                actual_best_val_acc_for_saving = epoch_val_accuracy
                best_model_state_dict_to_save = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                log_handler.log("SYSTEM",
                                f"New best validation accuracy for potential final save: {actual_best_val_acc_for_saving:.2f}% (Epoch {epoch + 1})")

            if epoch_val_accuracy > best_acc:
                best_acc = epoch_val_accuracy

            update_dashboard_contents(
                dashboard_components, training_metrics, log_handler, args, model,
                current_epoch=epoch + 1, max_epochs=args.max_epochs,
                batch_idx=len(train_loader), len_train_loader=len(train_loader)
            )

        total_training_time = time.time() - start_time
        log_handler.log(
            "SYSTEM",
            f"Training completed in {datetime.timedelta(seconds=int(total_training_time))}. "
            f"Best accuracy: {best_acc:.2f}%"
        )
    return best_acc, total_training_time, best_model_state_dict_to_save, actual_best_val_acc_for_saving


def _finalize_and_report_training(
        args: ArgsNamespace, model: CIFAR10Module, training_metrics: TrainingMetrics,
        best_acc_on_val: float, total_training_time: float, cifar_data: CIFARData,
        device: torch.device, log_handler: LogHandler,
        best_model_state_dict_to_save: Optional[dict], actual_best_val_acc_for_saving: float
):
    log_handler.log("SYSTEM", "Finalizing training and reporting...")

    final_checkpoint_path_msg = "No best model state was captured to save."
    if best_model_state_dict_to_save is not None:
        checkpoint_filename = args.checkpoint_name
        if checkpoint_filename == "none.pth" or not checkpoint_filename:
            noise_suffix = f"_{args.noise_type.value}" if args.noise_type != NoiseType.none else ""
            best_epoch_num = -1
            if training_metrics.val_acc_history:
                try:
                    best_idx = training_metrics.val_acc_history.index(actual_best_val_acc_for_saving / 100.0)
                    best_epoch_num = training_metrics.epochs[best_idx]
                    checkpoint_filename = f"{args.classifier}{noise_suffix}_best_epoch{best_epoch_num}_valacc{actual_best_val_acc_for_saving:.2f}.pth"
                except ValueError:
                    checkpoint_filename = f"{args.classifier}{noise_suffix}_best_valacc{actual_best_val_acc_for_saving:.2f}_final.pth"
            else:
                checkpoint_filename = f"{args.classifier}{noise_suffix}_best_valacc{actual_best_val_acc_for_saving:.2f}_final.pth"

        checkpoint_path = os.path.join("checkpoints", checkpoint_filename)
        log_handler.log("CHECKPOINT",
                        f"Saving best model (Val Acc: {actual_best_val_acc_for_saving:.2f}%) to {checkpoint_path} at end of training.")
        torch.save(best_model_state_dict_to_save, checkpoint_path)
        final_checkpoint_path_msg = f"checkpoints/{checkpoint_filename}"
    else:
        log_handler.log("CHECKPOINT", final_checkpoint_path_msg)

    log_handler.log("TESTING", "Starting final evaluation on the dedicated test set...")
    test_loader = cifar_data.test_dataloader()

    if best_model_state_dict_to_save:
        model.load_state_dict(best_model_state_dict_to_save)
        log_handler.log("MODEL", "Loaded best captured model state for final testing.")
    else:
        log_handler.log("MODEL", "No best model state captured; testing with the model from the last training epoch.")

    model.eval()
    test_loss_sum, test_correct, test_total = 0.0, 0, 0

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = model.criterion(outputs, targets)
            test_loss_sum += loss.item()
            correct_batch, total_batch = model.calculate_accuracy(outputs, targets)
            test_correct += correct_batch
            test_total += total_batch

    avg_test_loss = test_loss_sum / len(test_loader) if len(test_loader) > 0 else 0
    final_test_accuracy = 100. * test_correct / test_total if test_total > 0 else 0.0

    log_handler.log("TEST_RESULTS",
                    f"Final Test Set Performance -- Accuracy: {final_test_accuracy:.2f}%, Loss: {avg_test_loss:.4f}")

    log_handler.log("METRICS", "Measuring sharpness on the dedicated test set with the best model...")
    sharpness_data = training_metrics.measure_sharpness(
        model=model, criterion=model.criterion, val_loader=test_loader, device=device
    )
    log_handler.log("METRICS", f"Sharpness on test set: {sharpness_data}")

    training_metrics.final_metrics['test_accuracy'] = final_test_accuracy / 100.0
    training_metrics.final_metrics['test_loss'] = avg_test_loss
    training_metrics.save_final_metrics(
        best_acc=best_acc_on_val / 100.0, final_epoch=args.max_epochs, total_time=total_training_time
    )

    console.print(Panel(
        f"[bold green]Training completed![/bold green]\n"
        f"[bold]Model:[/bold] {args.classifier}\n"
        f"[bold]Total training time:[/bold] {datetime.timedelta(seconds=int(total_training_time))}\n"
        f"[bold]Best validation accuracy (during training):[/bold] {best_acc_on_val:.2f}%\n"
        f"[bold magenta]Final Test Set Accuracy:[/bold magenta] {final_test_accuracy:.2f}%\n"
        f"[bold]Best model checkpoint saved to (potentially):[/bold] {final_checkpoint_path_msg}\n"
        f"[bold]Metrics saved to:[/bold] {training_metrics.metrics_dir}",
        title="Training & Final Test Results",
        border_style="green"
    ))


def main(
        download_weights: bool = typer.Option(False, "--download-weights", "-w", help="Download pre-trained weights"),
        data_dir: str = "./data/cifar10",
        classifier: Classifier = typer.Option(Classifier.resnet18, "--classifier", "-c",
                                              help="Classifier model to use"),
        batch_size: int = typer.Option(128, "--batch-size", "-b", help="Batch size for training"),
        max_epochs: int = typer.Option(100, "--epochs", "-e", help="Maximum number of epochs"),
        num_workers: int = typer.Option(4, "--workers", "-n", help="Number of data loading workers"),
        subset: float = typer.Option(1.0, "--subset", "-s", help="Fraction of dataset to use for training (0.0-1.0)"),
        learning_rate: float = typer.Option(1e-2, "--lr", help="Initial learning rate"),
        weight_decay: float = typer.Option(1e-3, "--wd", help="Weight decay"),
        visualize_lr: bool = typer.Option(False, "--visualize-lr", help="Visualize learning rate schedule"),
        gpu_id: str = typer.Option("0", "--gpu", "-g", help="GPU ID(s) to use"),
        checkpoint_name: str = typer.Option("none.pth", "--save-as",
                                            help="Custom filename to save checkpoint (default: {classifier}_best.pth)"),
        noise_type: NoiseType = typer.Option(NoiseType.none, "--noise-type", "-nt",
                                             help="Type of noise regularization to apply"),
        noise_magnitude: float = typer.Option(0.01, "--noise-magnitude", "-nm", help="Initial magnitude of noise"),
        noise_schedule: NoiseSchedule = typer.Option(NoiseSchedule.constant, "--noise-schedule", "-ns",
                                                     help="Schedule for noise magnitude over time"),
        noise_layer: Optional[List[str]] = typer.Option(None, "--noise-layer", "-nl",
                                                        help="List of layer names to apply noise to. Example: --noise-layer conv1 --noise-layer fc"),
        noise_distribution: NoiseDistribution = typer.Option(NoiseDistribution.gaussian, "--noise-distribution", "-nd",
                                                             help="Distribution of noise (gaussian or uniform)"),
        optimizer: OptimizerChoice = typer.Option(OptimizerChoice.SGD, "--optimizer", "-o", help="Optimizer to use"),
        momentum: float = typer.Option(0.9, "--momentum", "-m", help="Momentum for SGD optimizer"),
        beta1: float = typer.Option(0.9, "--beta1", "-b1", help="Beta1 for Adam optimizer"),
        beta2: float = typer.Option(0.999, "--beta2", "-b2", help="Beta2 for Adam optimizer"),
        permanent: bool = typer.Option(False, "--permanent", "-p",
                                       help="Apply permanent noise to weights (default: False)"),
        noise_during_stuck_only: bool = typer.Option(False, "--noise-during-stuck-only", "-so",
                                                     help="Apply noise only when training is stuck (default: False)"),
        patience: int = typer.Option(5, "--patience",
                                     help="Number of epochs to wait before applying noise when training is stuck"),
        flag_window_size: int = typer.Option(5, "--flag-window-size",
                                             help="Number of epochs for history window for flags."),
        flag_min_epochs_to_check: int = typer.Option(10, "--flag-min-epochs", help="Min epochs before checking flags."),
        flag_overfitting_val_loss_epochs: int = typer.Option(3, "--flag-overfit-epochs",
                                                             help="Epochs val loss must increase for overfitting flag."),
        flag_plateau_min_delta: float = typer.Option(1e-4, "--flag-plateau-delta",
                                                     help="Min improvement to not be a plateau."),
        flag_grad_plateau_thr: float = typer.Option(0.05, "--flag-grad-plateau-thr",
                                                    help="Threshold for relative gradient norm improvement (e.g., 0.05 for 5%)."),

        flag_low_weight_update_threshold: float = typer.Option(1e-4, "--flag-low-weight-update-thr",
                                                               help="Threshold for low weight update norm flag."),
        disable_adaptive_flags: bool = typer.Option(False, "--disable-adaptive-flags",
                                                    help="Completely disable the adaptive flag system."),
        disable_graphs: bool = typer.Option(True, "--disable-graphs", help="Disable ASCII graphs in the dashboard."),

        relative_min_noise: float = typer.Option(0.01, "--relative-min-noise"),
        relative_max_noise: float = typer.Option(0.10, "--relative-max-noise"),
        consecutive_flag_trigger: int = typer.Option(3, "--consecutive-flag-trigger"),
        min_cooldown_epochs: int = typer.Option(5, "--min-cooldown-epochs"),
        max_cooldown_epochs: int = typer.Option(15, "--max-cooldown-epochs"),
        lr_restart_period: int = typer.Option(50, "--lr-restart-period"),

        detailed_metrics: bool = typer.Option(False, "--detailed-metrics",
                                              help="Enable slow, detailed per-class metric collection for the training set."),
):
    args_obj = _create_args_namespace(
        data_dir, checkpoint_name, download_weights, classifier, batch_size, max_epochs,
        num_workers, gpu_id, learning_rate, weight_decay, noise_type, noise_magnitude,
        noise_schedule, noise_layer, noise_distribution, optimizer, momentum, beta1, beta2,
        permanent, noise_during_stuck_only, patience, subset, visualize_lr,
        flag_window_size, flag_min_epochs_to_check, flag_overfitting_val_loss_epochs,
        flag_plateau_min_delta, flag_grad_plateau_thr, flag_low_weight_update_threshold,
        disable_adaptive_flags,
        disable_graphs,
        relative_min_noise,
        relative_max_noise,
        consecutive_flag_trigger,
        min_cooldown_epochs,
        max_cooldown_epochs,
        lr_restart_period,
        detailed_metrics,
    )
    class_compatible_args = ArgsNamespace(**vars(args_obj))
    class_compatible_args.classifier = args_obj.classifier.value

    training_metrics, log_handler = _initialize_training_components(class_compatible_args)
    device = _setup_environment(class_compatible_args, log_handler)

    if class_compatible_args.download_weights:
        log_handler.log("SYSTEM", "Downloading pre-trained weights...")
        CIFARData.download_weights()
        return

    cifar_data = CIFARData(class_compatible_args)
    cifar_data.prepare_data()
    cifar_data.setup()

    class_compatible_args.num_classes = cifar_data.num_classes
    class_compatible_args.dataset_name = cifar_data.dataset_name  # Also useful for logging

    class_compatible_args.cifar_data_module = cifar_data

    model, cifar_data = _prepare_data_and_model(class_compatible_args, device, log_handler)

    train_loader_len = len(cifar_data.train_dataloader())
    optimizer, scheduler = _configure_optimizer_and_scheduler(model, class_compatible_args, train_loader_len,
                                                              log_handler)

    best_val_acc, total_train_time, best_model_state_for_final_save, actual_best_val_acc = _run_training_loop(
        class_compatible_args, model, cifar_data, device, optimizer, scheduler,
        training_metrics, log_handler
    )

    _finalize_and_report_training(
        class_compatible_args, model, training_metrics, best_val_acc, total_train_time,
        cifar_data, device, log_handler, best_model_state_for_final_save, actual_best_val_acc
    )


if __name__ == "__main__":
    typer.run(main)
