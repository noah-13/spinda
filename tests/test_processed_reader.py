from __future__ import annotations

import json
from pathlib import Path

from hlv_toolkits.data import ProcessedJSONLReader
from hlv_toolkits.data.schemas import DiscoGeMMultiLevelSample, NLIDistributionSample, NLISample
from hlv_toolkits.data.serialization import sample_to_json_dict


def test_processed_jsonl_reader_round_trip(tmp_path: Path):
    samples = [
        NLISample(
            id="s1",
            task="nli",
            split="train",
            source="processed",
            premise="P1",
            hypothesis="H1",
            label=0,
        ),
        NLIDistributionSample(
            id="s2",
            task="nli",
            split="train",
            source="processed",
            premise="P2",
            hypothesis="H2",
            label=1,
            human_dist=[0.2, 0.7, 0.1],
        ),
        DiscoGeMMultiLevelSample(
            id="d1",
            task="discogem",
            split="train",
            source="processed",
            premise="P3",
            hypothesis="H3",
            hard_labels={"level1": 1, "level2": 2, "level3": 3},
            human_dists={"level1": [1.0, 0.0, 0.0, 0.0, 0.0], "level2": [0.0] * 17, "level3": [0.0] * 38},
        ),
    ]

    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(sample_to_json_dict(sample)) for sample in samples) + "\n",
        encoding="utf-8",
    )

    reader = ProcessedJSONLReader(data_path=str(data_dir))
    loaded = reader.load_train()

    assert len(loaded) == 3
    assert isinstance(loaded[0], NLISample)
    assert isinstance(loaded[1], NLIDistributionSample)
    assert isinstance(loaded[2], DiscoGeMMultiLevelSample)
    assert loaded[0].premise == "P1"
    assert loaded[1].human_dist == [0.2, 0.7, 0.1]
    assert loaded[2].hard_labels["level3"] == 3
