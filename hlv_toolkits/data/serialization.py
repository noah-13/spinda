from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Union

from hlv_toolkits.data.schemas import DiscoGeMMultiLevelSample, NLIDistributionSample, NLISample

UnifiedSample = Union[NLISample, NLIDistributionSample, DiscoGeMMultiLevelSample]


def sample_to_json_dict(sample: UnifiedSample) -> Dict[str, Any]:
    if not is_dataclass(sample):
        raise TypeError(f"Expected a dataclass sample, got {type(sample)!r}")
    payload = asdict(sample)
    payload["_schema"] = type(sample).__name__
    return payload


def sample_from_json_dict(payload: Dict[str, Any]) -> UnifiedSample:
    hard_labels = payload.get("hard_labels")
    human_dists = payload.get("human_dists")
    human_dist = payload.get("human_dist")

    base_kwargs = {
        "id": str(payload.get("id", "")),
        "task": str(payload.get("task", "nli")),
        "split": payload.get("split", "unknown"),
        "source": payload.get("source"),
        "meta": dict(payload.get("meta") or {}),
    }

    if hard_labels is not None or human_dists is not None:
        return DiscoGeMMultiLevelSample(
            premise=str(payload.get("premise", "")),
            hypothesis=str(payload.get("hypothesis", "")),
            hard_labels={str(k): int(v) for k, v in dict(hard_labels or {}).items()},
            human_dists={str(k): [float(x) for x in v] for k, v in dict(human_dists or {}).items()},
            **base_kwargs,
        )

    if human_dist is not None:
        return NLIDistributionSample(
            premise=str(payload.get("premise", "")),
            hypothesis=str(payload.get("hypothesis", "")),
            label=int(payload.get("label", -1)),
            human_dist=[float(x) for x in list(human_dist or [])],
            **base_kwargs,
        )

    return NLISample(
        premise=str(payload.get("premise", "")),
        hypothesis=str(payload.get("hypothesis", "")),
        label=int(payload.get("label", -1)),
        **base_kwargs,
    )


def dumps_sample(sample: UnifiedSample) -> str:
    return json.dumps(sample_to_json_dict(sample), ensure_ascii=False)


def dump_samples(samples: Iterable[UnifiedSample]) -> str:
    return "\n".join(dumps_sample(sample) for sample in samples)
