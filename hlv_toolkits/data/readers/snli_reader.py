from __future__ import annotations

from typing import List

from datasets import load_dataset

from hlv_toolkits.data.schemas import (
    NLISample,
    NLI_LABEL2ID,
    Split,
)
from hlv_toolkits.data.readers.base import NLISingleLabelReader


class SNLIReader(NLISingleLabelReader):
    """
    Reader for SNLI dataset.
    
    Loads SNLI from HuggingFace datasets library.
    Maps SNLI labels to standard label space:
    - entailment -> 0
    - neutral -> 1
    - contradiction -> 2
    """

    def __init__(self) -> None:
        super().__init__(task="nli")
        self.source = "snli"
        self._dataset_dict = None

    def load_split(self, split: Split) -> List[NLISample]:
        """
        Load SNLI data for the specified split.
        
        Args:
            split: One of "train", "dev" (validation), or "test"
            
        Returns:
            List of NLISample objects
        """
        # Map our split names to HuggingFace split names
        hf_split_map = {
            "train": "train",
            "dev": "validation",
            "valid": "validation",
            "validation": "validation",
            "test": "test",
        }
        
        if split not in hf_split_map:
            raise ValueError(f"Unknown split: {split}. Must be one of {list(hf_split_map.keys())}")
        
        hf_split = hf_split_map[split]
        
        # Load dataset from HuggingFace
        # Load full dataset dict first, then access the split
        if self._dataset_dict is None:
            self._dataset_dict = load_dataset("snli")
        dataset_dict = self._dataset_dict
        dataset = dataset_dict[hf_split]
        
        # Check if dataset is empty
        if len(dataset) == 0:
            return []
        
        samples = []
        for ex in dataset:
            # SNLI labels are:
            # - Integer: 0 (entailment), 1 (neutral), 2 (contradiction)
            label_raw = ex["label"]
            
            
            # Map label to integer ID
            if isinstance(label_raw, int):
                # SNLI uses integer labels: 0=entailment, 1=neutral, 2=contradiction
                # This matches our label space, so use directly
                if label_raw not in [0, 1, 2]:
                    continue
                label_id = label_raw
            elif isinstance(label_raw, str):
                # String label, map using dictionary
                if label_raw not in NLI_LABEL2ID:
                    continue
                label_id = NLI_LABEL2ID[label_raw]
            else:
                continue
            
            sample = NLISample(
                id=str(ex.get("pairID", len(samples))),
                task=self.task,
                split=split,
                source=self.source,
                premise=ex["premise"],
                hypothesis=ex["hypothesis"],
                label=label_id,
            )
            samples.append(sample)
        
        return samples
