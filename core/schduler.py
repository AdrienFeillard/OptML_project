import math
import warnings
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineLR(_LRScheduler):
    """
    Sets the learning rate of each parameter group to follow a linear warmup schedule
    between warmup_start_lr and base_lr followed by a cosine annealing schedule between
    base_lr and eta_min.
    .. warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.
    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_epochs (int): Maximum number of iterations for linear warmup
        max_epochs (int): Maximum number of iterations
        warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
    Example:
        >>> layer = nn.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
        >>> #
        >>> # the default case
        >>> for epoch in range(40):
        ...     # train(...)
        ...     # validate(...)
        ...     scheduler.step()
        >>> #
        >>> # passing epoch param case
        >>> for epoch in range(40):
        ...     scheduler.step(epoch)
        ...     # train(...)
        ...     # validate(...)
    """

    def __init__(
            self,
            optimizer: Optimizer,
            warmup_epochs: int,
            max_epochs: int,
            warmup_start_lr: float = 1e-8,
            eta_min: float = 1e-8,
            last_epoch: int = -1,
    ) -> None:

        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"]
                + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (
                2 * (self.max_epochs - self.warmup_epochs)
        ) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs)))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (
                    1
                    + math.cos(
                math.pi
                * (self.last_epoch - self.warmup_epochs)
                / (self.max_epochs - self.warmup_epochs)
            )
            )
            / (
                    1
                    + math.cos(
                math.pi
                * (self.last_epoch - self.warmup_epochs - 1)
                / (self.max_epochs - self.warmup_epochs)
            )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr
                + self.last_epoch
                * (base_lr - self.warmup_start_lr)
                / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (
                    1
                    + math.cos(
                math.pi
                * (self.last_epoch - self.warmup_epochs)
                / (self.max_epochs - self.warmup_epochs)
            )
            )
            for base_lr in self.base_lrs
        ]

class CosineAnnealingWithDecayingRestartsLR(_LRScheduler):
    """
    Custom scheduler that implements cosine annealing with warm restarts, where the
    maximum learning rate decays linearly over the entire training duration to a
    specified final value.
    """

    def __init__(
            self,
            optimizer: Optimizer,
            T_0: int,
            total_steps: int,
            eta_min: float = 0,
            final_max_lr: float = 1e-5, # MODIFIED: Use an absolute final LR
            T_mult: int = 1,
            last_epoch: int = -1,
    ):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            T_0 (int): Number of steps in the first restart cycle.
            total_steps (int): Total number of steps in the entire training process.
                               Used to calculate the decay of the max LR.
            eta_min (float): Minimum learning rate. Default: 0.
            final_max_lr (float): The absolute maximum learning rate for the final restart cycle.
            T_mult (int): A factor that increases T_i after a restart. Default: 1.
            last_epoch (int): The index of the last epoch. Default: -1.
        """
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))

        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.total_steps = total_steps
        self.final_max_lr = final_max_lr # MODIFIED: Store the new parameter
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        new_lrs = []
        for base_lr in self.base_lrs:
            # --- MODIFIED: Calculate the current (decaying) maximum learning rate ---
            progress = self.last_epoch / self.total_steps if self.total_steps > 0 else 1

            # Linearly decay the max LR from the initial base_lr down to the specified final_max_lr
            current_max_lr = base_lr - progress * (base_lr - self.final_max_lr)

            # --- Cosine annealing formula remains the same, but uses the new current_max_lr ---
            term1 = self.eta_min
            term2_numerator = current_max_lr - self.eta_min
            term2_denominator = 2
            term2_multiplier = 1 + math.cos(math.pi * self.T_cur / self.T_i) if self.T_i > 0 else 0

            lr = term1 + (term2_numerator / term2_denominator) * term2_multiplier
            new_lrs.append(lr)

        return new_lrs

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                epoch = 0
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.T_i = self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:
            def __init__(self, sched):
                self.sched = sched
            def __enter__(self):
                self.sched._get_lr_called_within_step = True
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.sched._get_lr_called_within_step = False

        with _enable_get_lr_call(self):
            for i, (param_group, lr) in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]