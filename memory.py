import json
from dataclasses import dataclass, asdict
from pathlib import Path

DEFAULT_PATH = Path("memory.json")


@dataclass
class Memory:
    name: str | None = None
    awaiting_name: bool = False
    path: Path = DEFAULT_PATH

    def load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.name = data.get("name")
        self.awaiting_name = data.get("awaiting_name", False)

    def save(self) -> None:
        data = {
            "name": self.name,
            "awaiting_name": self.awaiting_name,
        }
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")