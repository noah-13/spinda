from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Protocol, runtime_checkable

from spinda.data.schemas import (
    AnySample,
    Split,
)


@runtime_checkable
class SupportsLen(Protocol):
    def __len__(self) -> int: ...


class BaseReader(ABC):
    """
    Abstract base class for data readers.

    Convention:
    - Focus on converting external data => standardized Sample objects
    - Do not handle tokenization / dataloader tasks
    """

    task: str

    def __init__(self, task: str = "nli") -> None:
        self.task = task

    @abstractmethod
    def load_split(self, split: Split) -> List[AnySample]:
        """
        Load samples for the specified split.
        """

    # Convenient alias methods
    def load_train(self) -> List[AnySample]:
        return self.load_split("train")

    def load_dev(self) -> List[AnySample]:
        # Allow different datasets to use dev / validation / valid
        return self.load_split("dev")

    def load_test(self) -> List[AnySample]:
        return self.load_split("test")
