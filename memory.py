import json
from dataclasses import dataclass, asdict
from pathlib import Path

MEMORY_FILE = Path("memory.json")

@dataclass
class Memory:
    name: str | None = None
    awaiting_name: bool = False

    @classmethod
    def load(cls) -> None:
        if not MEMORY_FILE.exists():
            return cls()

        try:
            data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
            return cls(**data)
        except Exception:
            return cls()

    def save(self) -> None:
        MEMORY_FILE.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
