"""Tests for data readers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hlv_toolkits.data import ChaosNLIReader, SNLIReader
from hlv_toolkits.data.schemas import NLIDistributionSample, NLISample


def test_snli_reader_loads():
    reader = SNLIReader()
    assert reader.task == "nli"
    assert reader.source == "snli"


def test_snli_reader_load_split_monkeypatched(monkeypatch: pytest.MonkeyPatch):
    # Avoid any network/cache dependency from HuggingFace datasets.
    import hlv_toolkits.data.readers.snli_reader as snli_reader_module

    calls = {"n": 0}

    def fake_load_dataset(name: str):
        assert name == "snli"
        calls["n"] += 1
        return {
            "train": [
                {"pairID": "t1", "premise": "P1", "hypothesis": "H1", "label": 0},
                {"pairID": "t2", "premise": "P2", "hypothesis": "H2", "label": "neutral"},
                {"pairID": "t3", "premise": "P3", "hypothesis": "H3", "label": -1},  # filtered out
            ],
            "validation": [
                {"pairID": "v1", "premise": "P4", "hypothesis": "H4", "label": 2},
            ],
            "test": [
                {"pairID": "s1", "premise": "P5", "hypothesis": "H5", "label": 1},
            ],
        }

    monkeypatch.setattr(snli_reader_module, "load_dataset", fake_load_dataset)

    reader = SNLIReader()

    train = reader.load_train()
    assert [s.id for s in train] == ["t1", "t2"]
    assert all(isinstance(s, NLISample) for s in train)
    assert all(s.task == "nli" for s in train)
    assert all(s.source == "snli" for s in train)
    assert all(s.split == "train" for s in train)
    assert {s.label for s in train} == {0, 1}

    dev = reader.load_dev()
    assert len(dev) == 1
    assert dev[0].split == "dev"
    assert dev[0].label == 2

    test = reader.load_test()
    assert len(test) == 1
    assert test[0].split == "test"
    assert test[0].label == 1
    assert calls["n"] == 1


def test_snli_reader_unknown_split_raises():
    reader = SNLIReader()
    with pytest.raises(ValueError):
        reader.load_split("unknown")  # type: ignore[arg-type]


def test_chaosnli_reader_defaults_to_repo_file_if_present():
    repo_file = Path("data/external/chaosnli/chaosNLI_snli.jsonl")
    reader = ChaosNLIReader(data_path=None, source="chaosnli-snli")
    if repo_file.exists():
        samples = reader.load_dev()
        assert len(samples) > 0
        assert samples[0].premise
        assert samples[0].hypothesis
        assert len(samples[0].human_dist) == 3
        assert abs(sum(samples[0].human_dist) - 1.0) < 1e-6
    else:
        with pytest.raises(ValueError):
            reader.load_dev()


def test_chaosnli_reader_missing_file_raises(tmp_path: Path):
    missing = tmp_path / "missing.jsonl"
    reader = ChaosNLIReader(data_path=str(missing))
    with pytest.raises(FileNotFoundError):
        reader.load_dev()


def test_chaosnli_reader_parses_supported_formats(tmp_path: Path):
    p = tmp_path / "mini_chaosnli.jsonl"
    examples = [
        # Toolkits-style counts
        {
            "pairID": "p1",
            "premise": "A woman is talking on the phone while standing next to a dog.",
            "hypothesis": "A woman is walking her dog.",
            "label_dist": [4, 67, 29],
        },
        # Official ChaosNLI-style counts + nested example
        {
            "uid": "u1",
            "label_count": [30, 70, 0],
            "example": {"premise": "P", "hypothesis": "H"},
        },
        # Official ChaosNLI-style probabilities (already normalized)
        {
            "uid": "u2",
            "label_dist": [0.03, 0.94, 0.03],
            "example": {"premise": "P2", "hypothesis": "H2"},
        },
        # Official ChaosNLI-style label_counter map
        {
            "uid": "u3",
            "label_counter": {"e": 10, "n": 20, "c": 70},
            "example": {"premise": "P3", "hypothesis": "H3"},
        },
    ]
    p.write_text("\n".join(json.dumps(ex) for ex in examples) + "\n", encoding="utf-8")

    reader = ChaosNLIReader(data_path=str(p))
    samples = reader.load_split("dev")

    assert len(samples) == 4
    assert all(isinstance(s, NLIDistributionSample) for s in samples)
    assert all(s.task == "nli" for s in samples)
    assert all(s.source == "chaosnli-snli" for s in samples)
    assert all(s.split == "dev" for s in samples)
    assert all(len(s.human_dist) == 3 for s in samples)
    assert all(abs(sum(s.human_dist) - 1.0) < 1e-6 for s in samples)
    assert samples[0].id == "p1"
    assert samples[1].id == "u1"
    assert samples[2].id == "u2"
    assert samples[3].id == "u3"


def test_chaosnli_reader_loads_repo_jsonl_if_present():
    repo_file = Path("data/external/chaosnli/chaosNLI_snli.jsonl")
    if not repo_file.exists():
        pytest.skip("Repo ChaosNLI file not available in this checkout.")

    reader = ChaosNLIReader(data_path=str(repo_file), source="chaosnli-snli")
    samples = reader.load_dev()
    assert len(samples) > 0
    assert samples[0].premise
    assert samples[0].hypothesis
    assert len(samples[0].human_dist) == 3
    assert abs(sum(samples[0].human_dist) - 1.0) < 1e-6


def test_chaosnli_reader_alpha_nli_is_rejected_if_present():
    repo_file = Path("data/external/chaosnli/chaosNLI_alphanli.jsonl")
    if not repo_file.exists():
        pytest.skip("Repo ChaosNLI alphaNLI file not available in this checkout.")

    reader = ChaosNLIReader(data_path=str(repo_file), source="chaosnli-alphanli")
    with pytest.raises(ValueError, match="alphaNLI|2-way|2-way distribution"):
        reader.load_dev()


def test_chaosnli_reader_loads_mnli_if_present():
    repo_file = Path("data/external/chaosnli/chaosNLI_mnli_m.jsonl")
    if not repo_file.exists():
        pytest.skip("Repo ChaosNLI MNLI file not available in this checkout.")

    reader = ChaosNLIReader(data_path=str(repo_file), source="chaosnli-mnli")
    samples = reader.load_dev()
    assert len(samples) > 0
    assert samples[0].premise
    assert samples[0].hypothesis
    assert len(samples[0].human_dist) == 3


def test_discogem_reader_2_0_soft_and_hard(tmp_path: Path):
    p = tmp_path / "discogem20.tsv"
    p.write_text(
        "\t".join([
            "split",
            "itemid",
            "arg1_context_en",
            "arg2_context_en",
            "WAWA_en",
            "WAWA_dist_en",
        ])
        + "\n"
        + "\t".join([
            "train",
            "id1",
            "A1",
            "A2",
            "cause",
            "{'cause': 0.7, 'contrast': 0.3}",
        ])
        + "\n",
        encoding="utf-8",
    )

    from hlv_toolkits.data import DiscoGeMReader

    soft_reader = DiscoGeMReader(data_path=str(p), version="2.0", label_mode="soft")
    soft = soft_reader.load_train()
    assert len(soft) == 1
    assert soft[0].task == "discogem"
    assert soft[0].premise == "A1"
    assert soft[0].hypothesis == "A2"
    assert abs(sum(soft[0].human_dist) - 1.0) < 1e-6

    hard_reader = DiscoGeMReader(data_path=str(p), version="2.0", label_mode="hard")
    hard = hard_reader.load_train()
    assert len(hard) == 1
    assert hard[0].human_dist == []
    assert hard[0].label >= 0


def test_discogem_reader_auto_version_detects_2_0(tmp_path: Path):
    p = tmp_path / "discogem_auto.tsv"
    p.write_text(
        "split\titemid\targ1_context_en\targ2_context_en\tWAWA_en\tWAWA_dist_en\n"
        "train\ta1\tP\tH\tcause\t{'cause': 1.0}\n",
        encoding="utf-8",
    )

    from hlv_toolkits.data import DiscoGeMReader

    reader = DiscoGeMReader(data_path=str(p), version="auto", label_mode="soft")
    samples = reader.load_train()
    assert len(samples) == 1
    assert samples[0].id == "a1"
    assert samples[0].premise == "P"
    assert samples[0].hypothesis == "H"
