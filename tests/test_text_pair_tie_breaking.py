import json

from hlv_toolkits.data import TextPairClassificationJSONLReader


def test_vote_distribution_labels_break_ties_reproducibly_without_lowest_label_bias(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "dataset.json").write_text(
        json.dumps(
            {
                "format": "text_pair_label_distribution",
                "label_mode": "soft",
                "labels": ["zero", "one"],
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {"id": f"tie-{index}", "text_a": "a", "text_b": "b", "annotation_labels": [0, 1]}
        for index in range(20)
    ]
    for split in ("train", "dev", "test"):
        (data / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    first = TextPairClassificationJSONLReader(str(data), label_mode="soft_to_hard").load_train()
    second = TextPairClassificationJSONLReader(str(data), label_mode="soft_to_hard").load_train()
    soft_test = TextPairClassificationJSONLReader(str(data)).load_test()

    assert [sample.label for sample in first] == [sample.label for sample in second]
    assert {sample.label for sample in first} == {0, 1}
    assert [sample.label for sample in soft_test] == [sample.label for sample in first]
    assert all(sample.human_dist == [0.5, 0.5] for sample in first)
