import os
import zipfile

import numpy as np
import requests
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10,CIFAR100
from tqdm import tqdm
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
from rich.panel import Panel

from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit

# Initialize Rich console
console = Console()

class CIFARData:
    def __init__(self, args):
        self.args = args
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)

        # Store configuration
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        if 'cifar100' in self.data_dir.lower():
            print("Inferred CIFAR-100 from data directory path.")
            self.dataset_name = "CIFAR-100"
            self.dataset_class = CIFAR100
            self.num_classes = 100
        else:
            print("Defaulting to CIFAR-10 based on data directory path.")
            self.dataset_name = "CIFAR-10"
            self.dataset_class = CIFAR10
            self.num_classes = 10

        # Datasets will be initialized in setup
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset_final = None
        # Store the % of the dataset we want to use
        self.subset = args.subset

        self.classes = []

    @staticmethod
    def download_weights():
        url = (
            "https://rutgers.box.com/shared/static/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip"
        )

        # Using Rich for a better download progress bar
        console.print(Panel("[bold cyan]Downloading pre-trained weights...[/bold cyan]",
                            title="Download", border_style="cyan"))

        with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40, complete_style="cyan"),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task("[green]Downloading weights", start=False)

            # Prepare a session for the request
            session = requests.Session()
            response = session.get(url, stream=True)

            # Get content length if available
            total_size = int(response.headers.get("content-length", 0))
            progress.update(task_id, total=total_size)
            progress.start_task(task_id)

            # Download and write to file
            with open("state_dicts.zip", "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress.update(task_id, advance=len(chunk))

        if total_size == 0:
            console.print("[yellow]Warning: Couldn't determine file size.[/yellow]")

        console.print("[bold green]Download successful![/bold green] Unzipping file...")
        path_to_zip_file = os.path.join(os.getcwd(), "state_dicts.zip")
        directory_to_extract_to = os.path.join(os.getcwd(), "cifar10_models")

        # Create directory if it doesn't exist
        os.makedirs(directory_to_extract_to, exist_ok=True)

        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            # Count files for progress
            file_count = len(zip_ref.infolist())

            with Progress(
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(bar_width=40, complete_style="green"),
                    TextColumn("[bold green]{task.completed}/{task.total}"),
                    TimeRemainingColumn(),
            ) as progress:
                extract_task = progress.add_task("[cyan]Extracting files", total=file_count)

                for i, file in enumerate(zip_ref.infolist()):
                    zip_ref.extract(file, directory_to_extract_to)
                    progress.update(extract_task, advance=1)

        console.print(Panel("[bold green]Pre-trained weights downloaded and extracted successfully![/bold green]",
                            title="Download Complete", border_style="green"))

    def prepare_data(self):
        os.makedirs(self.data_dir, exist_ok=True)

        # --- CHANGE 3: Make print statements and function calls dynamic
        console.print(Panel(f"[bold cyan]Preparing {self.dataset_name} dataset...[/bold cyan]",
                            title="Dataset Preparation", border_style="cyan"))
        try:
            # Use the dynamic self.dataset_class instead of hardcoded CIFAR10
            self.dataset_class(root=self.data_dir, train=True, download=True)
            self.dataset_class(root=self.data_dir, train=False, download=True)
            console.print(f"[bold green]{self.dataset_name} dataset ready![/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error downloading dataset: {e}[/bold red]")
            console.print("[yellow]Please check your internet connection and try again.[/yellow]")
            raise

    def setup(self):
        console.print(Panel("[bold cyan]Setting up data transformations and loaders...[/bold cyan]",
                            title="Data Setup", border_style="cyan"))

        # Define transforms
        train_transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        test_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )

        # Log transforms
        console.print("[bold]Training transforms:[/bold]")
        console.print("  • RandomCrop(32, padding=4)")
        console.print("  • RandomHorizontalFlip()")
        console.print("  • ToTensor()")
        console.print(f"  • Normalize(mean={self.mean}, std={self.std})")

        console.print("[bold]Test transforms:[/bold]")
        console.print("  • ToTensor()")
        console.print(f"  • Normalize(mean={self.mean}, std={self.std})")

        # Load datasets
        with console.status("[cyan]Loading training dataset...[/cyan]"):
            self.train_dataset = self.dataset_class(
                root=self.data_dir, train=True, transform=train_transform
            )

        with console.status("[cyan]Loading and splitting original test set for validation and final test...[/cyan]"):
            original_test_set = self.dataset_class(
                root=self.data_dir, train=False, transform=test_transform
            )

        original_test_targets = np.array(original_test_set.targets)

        if len(original_test_set) > 0:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
            val_indices, test_final_indices = next(splitter.split(np.zeros(len(original_test_targets)), original_test_targets))

            self.val_dataset = Subset(original_test_set, val_indices)
            self.test_dataset_final = Subset(original_test_set, test_final_indices)
        else:
            self.val_dataset = original_test_set
            self.test_dataset_final = original_test_set

        console.print(f"[green]Training samples: {len(self.train_dataset)}[/green]")
        console.print(f"[green]Validation samples (for training): {len(self.val_dataset)}[/green]")
        console.print(f"[green]Final Test samples (held-out): {len(self.test_dataset_final)}[/green]")

        # Dynamically get and print class information
        self.classes = self.train_dataset.classes
        console.print(f"[bold]{self.dataset_name} Classes:[/bold]")
        console.print(f"  • Total classes: [cyan]{len(self.classes)}[/cyan]")
        for i, cls in enumerate(self.classes):
                console.print(f"  • Class {i}: [cyan]{cls}[/cyan]")

    def train_dataloader(self):
        if self.subset < 1.0:
            targets = np.array(self.train_dataset.targets)
            sss = StratifiedShuffleSplit(n_splits=1, train_size=self.subset, random_state=42)
            indices, _ = next(sss.split(np.zeros(len(targets)), targets))
            dataset = Subset(self.train_dataset, indices)
        else:
            dataset = self.train_dataset

        class_counts = {i: 0 for i in range(self.num_classes)}

        # Check if dataset has targets attribute directly
        # The Subset wrapper does not expose targets directly, we need to access the underlying dataset
        targets_to_iterate = []
        if isinstance(dataset, Subset):
            targets_to_iterate = [dataset.dataset.targets[i] for i in dataset.indices]
        else:
            targets_to_iterate = dataset.targets

        for label in targets_to_iterate:
            class_counts[label] += 1

        print('Number of classes per label in training subset:', class_counts)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )


    def val_dataloader(self):
        # This now uses the new, smaller self.val_dataset
        return DataLoader( #
            self.val_dataset, # Uses the NEW validation set
            batch_size=self.batch_size, #
            num_workers=self.num_workers, #
            # drop_last=True, # Usually False for val/test to evaluate all samples
            drop_last=False,
            pin_memory=True, #
            persistent_workers=True if self.num_workers > 0 else False, #
        )

    def test_dataloader(self):
        # This now returns a DataLoader for the new, final held-out test set
        if self.test_dataset_final is None:
            console.print("[bold red]Error: Final test dataset not setup![/bold red]")
            # Fallback or raise error
            return self.val_dataloader() # Fallback to val_dataloader to prevent crash, but this isn't ideal

        return DataLoader(
            self.test_dataset_final, # Use the new final test set
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False, # Important for test set
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )