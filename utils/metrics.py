import torch
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