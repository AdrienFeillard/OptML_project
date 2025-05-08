import time
from rich.table import Table
from rich import box

class LogHandler:
    """Simple logging system with timestamps"""
    def __init__(self, max_entries=100):
        self.max_entries = max_entries
        self.logs = []

    def log(self, source, message, level="INFO"):
        """Add a log entry with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        entry = {
            "time": timestamp,
            "source": source.upper(),
            "level": level.upper(),
            "message": message
        }
        self.logs.append(entry)

        # Trim if exceeding max entries
        if len(self.logs) > self.max_entries:
            self.logs = self.logs[-self.max_entries:]

    def get_logs_table(self):
        """Return a Rich table of log entries"""
        log_table = Table(show_header=True, box=box.SIMPLE, expand=True)
        log_table.add_column("Time", style="dim")
        log_table.add_column("Source", style="yellow")
        log_table.add_column("Message", style="white", ratio=3)

        for entry in self.logs:
            source_style = "yellow"
            if entry["level"] == "ERROR":
                source_style = "red"
            elif entry["level"] == "WARNING":
                source_style = "orange3"

            log_table.add_row(
                entry["time"],
                f"[{source_style}]{entry['source']}[/{source_style}]",
                entry["message"]
            )

        return log_table