from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from hlv_toolkits.data.schemas import (
    NLIDistributionSample,
    NLI_LABEL2ID,
    NLI_NUM_LABELS,
    Split,
)
from hlv_toolkits.data.readers.base import NLIDistributionReader


class ChaosNLIReader(NLIDistributionReader):
    """
    Reader for ChaosNLI dataset.
    
    ChaosNLI contains human annotation distributions (100 annotators per instance).
    Each instance has a distribution over the 3 NLI labels.
    
    Supported JSONL formats include:
    - Toolkits-style:
      {
        "pairID": "...",
        "premise": "...",
        "hypothesis": "...",
        "label_dist": [count_entailment, count_neutral, count_contradiction]
      }
    - Official ChaosNLI v1.0-style:
      {
        "uid": "...",
        "label_count": [entailment_count, neutral_count, contradiction_count],
        "label_dist": [p_entailment, p_neutral, p_contradiction],
        "example": {"premise": "...", "hypothesis": "...", ...},
        "label_counter": {"e": 4, "n": 67, "c": 29}
      }
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        source: str = "chaosnli-snli",
    ) -> None:
        """
        Initialize ChaosNLI reader.
        
        Args:
            data_path: Path to ChaosNLI JSONL file. If None, will try to load
                      from default location or raise error.
            source: Source identifier for the samples (e.g., "chaosnli-snli", "chaosnli-mnli")
        """
        super().__init__(task="nli")
        self.source = source
        self.data_path = Path(data_path) if data_path else None

    def _infer_default_data_path(self) -> Path:
        source = (self.source or "").lower()
        if "alpha" in source:
            filename = "chaosNLI_alphanli.jsonl"
        elif "mnli" in source:
            filename = "chaosNLI_mnli_m.jsonl"
        else:
            filename = "chaosNLI_snli.jsonl"

        return Path("data/external/chaosnli") / filename

    def load_split(self, split: Split) -> List[NLIDistributionSample]:
        """
        Load ChaosNLI data for the specified split.
        
        Note: ChaosNLI may not have explicit train/dev/test splits.
        This method loads all data and filters by split if available,
        or returns all data if split information is not present.
        
        Args:
            split: Split name (may be ignored if dataset doesn't have splits)
            
        Returns:
            List of NLIDistributionSample objects with human_dist populated
        """
        data_path = self.data_path
        if data_path is None:
            data_path = self._infer_default_data_path()
            if not data_path.exists():
                raise ValueError(
                    "data_path must be provided (or place the official ChaosNLI v1.0 files under "
                    "`data/external/chaosnli`). "
                    "Download link: https://www.dropbox.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip"
                )
        
        if not data_path.exists():
            raise FileNotFoundError(f"ChaosNLI data file not found: {data_path}")
        
        samples = []
        
        # Read JSONL file
        with open(data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num}: {e}")
                
                # Extract human distribution
                human_dist = self._extract_human_dist(ex)
                if human_dist is None:
                    continue

                if len(human_dist) != NLI_NUM_LABELS:
                    if len(human_dist) == 2:
                        raise ValueError(
                            "This looks like ChaosNLI-alphaNLI (2-way distribution), which is not supported by "
                            "ChaosNLIReader (expects 3-way NLI: entailment/neutral/contradiction)."
                        )
                    continue
                
                total = float(sum(human_dist))
                if total <= 0:
                    continue
                if abs(total - 1.0) > 1e-6:
                    human_dist = [float(x) / total for x in human_dist]
                
                # Get label (majority vote for compatibility)
                label_id = max(range(len(human_dist)), key=lambda i: human_dist[i])

                ex_text = ex.get("example") if isinstance(ex.get("example"), dict) else {}
                premise = ex.get("premise") or ex_text.get("premise", "")
                hypothesis = ex.get("hypothesis") or ex_text.get("hypothesis", "")

                sample_id = ex.get("pairID") or ex.get("uid") or ex_text.get("uid") or f"chaosnli_{line_num}"
                
                sample = NLIDistributionSample(
                    id=str(sample_id),
                    task=self.task,
                    split=split,  # Use provided split
                    source=self.source,
                    premise=premise,
                    hypothesis=hypothesis,
                    label=label_id,
                    human_dist=[float(x) for x in human_dist],
                )
                samples.append(sample)
        
        return samples

    def _extract_human_dist(self, ex: Dict) -> Optional[List[float]]:
        """
        Extract human label distribution from example.
        
        Supports multiple formats:
        1. "label_count": [count_entailment, count_neutral, count_contradiction]
        2. "label_dist": [p_entailment, p_neutral, p_contradiction] OR counts in the same order
        3. "label_counter": {"e": 4, "n": 67, "c": 29} or {"entailment": 4, ...}
        4. "annotator_labels": ["entailment", "neutral", ...] (list of label strings)
        
        Returns:
            List of floats in label order [entailment, neutral, contradiction]
            or None if format is not recognized
        """
        # Prefer label_counter when present (it's the most explicit and always disambiguates labels).
        if "label_counter" in ex and isinstance(ex["label_counter"], dict):
            counter = ex["label_counter"]
            if any(k in counter for k in ("e", "n", "c")):
                return [
                    float(counter.get("e", 0)),
                    float(counter.get("n", 0)),
                    float(counter.get("c", 0)),
                ]
            if any(k in counter for k in ("1", "2", 1, 2)):
                return [
                    float(counter.get("1", counter.get(1, 0))),
                    float(counter.get("2", counter.get(2, 0))),
                ]
            counts = [0.0, 0.0, 0.0]
            for k, v in counter.items():
                if k in NLI_LABEL2ID:
                    counts[NLI_LABEL2ID[k]] = float(v)
            if sum(counts) > 0:
                return counts

        # Official ChaosNLI v1.0: explicit counts
        if "label_count" in ex:
            counts = ex["label_count"]
            if isinstance(counts, list) and len(counts) == NLI_NUM_LABELS:
                return [float(x) for x in counts]
            if isinstance(counts, list) and len(counts) == 2:
                return [float(x) for x in counts]

        # Format 1/2: distribution (may be counts or probabilities)
        if "label_dist" in ex:
            dist = ex["label_dist"]
            if isinstance(dist, list) and len(dist) == NLI_NUM_LABELS:
                return [float(x) for x in dist]
            if isinstance(dist, list) and len(dist) == 2:
                return [float(x) for x in dist]
        
        # Format 2: List of annotator labels
        if "annotator_labels" in ex:
            labels = ex["annotator_labels"]
            if isinstance(labels, list):
                counts = [0.0, 0.0, 0.0]
                for label_str in labels:
                    if label_str in ("e", "n", "c"):
                        label_id = {"e": 0, "n": 1, "c": 2}[label_str]
                        counts[label_id] += 1.0
                    elif label_str in NLI_LABEL2ID:
                        label_id = NLI_LABEL2ID[label_str]
                        counts[label_id] += 1.0
                return counts
        
        return None
