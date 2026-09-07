from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional
from hlv_toolkits.data.readers.base import BaseReader
from hlv_toolkits.data.schemas import MultilevelSample, Split

class TextPairMultilevelJSONLReader(BaseReader):
    def __init__(self, data_path: Optional[str] = None, task: str = "discogem", **_: object) -> None:
        super().__init__(task=task)
        self.data_path = Path(data_path) if data_path else None
    def load_split(self, split: Split) -> List[MultilevelSample]:
        if self.data_path is None: raise ValueError("data_path is required")
        path = self.data_path / f"{split}.jsonl" if self.data_path.is_dir() else self.data_path
        rows: List[MultilevelSample] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            p = json.loads(line)
            row_split = "dev" if p.get("split") in {"valid", "validation"} else p.get("split", split)
            target = "dev" if split in {"valid", "validation"} else split
            if row_split != target: continue
            rows.append(MultilevelSample(id=str(p["id"]), task=p.get("task", self.task), split=target, source=p.get("source"), meta=dict(p.get("meta") or {}), premise=str(p.get("text_a", p.get("premise", ""))), hypothesis=str(p.get("text_b", p.get("hypothesis", ""))), hard_labels={str(k): int(v) for k,v in dict(p.get("hard_labels") or {}).items()}, human_dists={str(k): [float(x) for x in v] for k,v in dict(p.get("human_dists") or {}).items()}))
        return rows
