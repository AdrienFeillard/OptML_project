import os
import zipfile

import requests
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
from rich.panel import Panel

from torch.utils.data import Subset

# Initialize Rich console
console = Console()

class CIFAR10Data:
    def __init__(self, args):
        self.args = args
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)

        # Store configuration
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        # Datasets will be initialized in setup
        self.train_dataset = None
        self.val_dataset = None

        # Store the % of the dataset we want to use
        self.subset = args.subset

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
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)

        # Always use download=True to be safe
        console.print(Panel("[bold cyan]Preparing CIFAR-10 dataset...[/bold cyan]",
                            title="Dataset Preparation", border_style="cyan"))
        try:
            CIFAR10(root=self.data_dir, train=True, download=True)
            CIFAR10(root=self.data_dir, train=False, download=True)
            console.print("[bold green]CIFAR-10 dataset ready![/bold green]")
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
            self.train_dataset = CIFAR10(
                root=self.data_dir, train=True, transform=train_transform
            )

        with console.status("[cyan]Loading validation dataset...[/cyan]"):
            self.val_dataset = CIFAR10(
                root=self.data_dir, train=False, transform=test_transform
            )

        console.print(f"[green]Training samples: {len(self.train_dataset)}[/green]")
        console.print(f"[green]Validation samples: {len(self.val_dataset)}[/green]")

        # Print dataset class information
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        console.print("[bold]CIFAR-10 Classes:[/bold]")
        for i, cls in enumerate(classes):
            console.print(f"  • Class {i}: [cyan]{cls}[/cyan]")

    def train_dataloader(self):
        if self.subset < 1.0:
            subset_size = int(len(self.train_dataset) * self.subset)
            indices = torch.randperm(len(self.train_dataset)).tolist()[:subset_size]
            subset_train_dataset = Subset(self.train_dataset, indices)
            dataset = subset_train_dataset
        else:
            dataset = self.train_dataset

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
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return self.val_dataloader()