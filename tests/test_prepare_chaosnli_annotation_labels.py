from spinda.scripts.prepare_chaosnli_annotation_labels import _split_kfold_train_dev_test


def _row(identifier: str, text_a: str, text_b: str) -> dict[str, str]:
    return {"id": identifier, "text_a": text_a, "text_b": text_b}


def test_kfold_split_matches_train_eval_hlv_protocol_and_keeps_pairs_together():
    # The two rows for pair-0 represent duplicate annotations of one input.
    rows = [_row("pair-0-a", "premise 0", "hypothesis 0"), _row("pair-0-b", "premise 0", "hypothesis 0")]
    rows.extend(_row(f"pair-{index}", f"premise {index}", f"hypothesis {index}") for index in range(1, 20))

    folds = _split_kfold_train_dev_test(rows, folds=5, dev_portion=0.1, seed=42)

    assert len(folds) == 5
    all_test_ids = set()
    for splits in folds:
        split_ids = {name: {row["id"] for row in values} for name, values in splits.items()}
        assert not (split_ids["train"] & split_ids["dev"])
        assert not (split_ids["train"] & split_ids["test"])
        assert not (split_ids["dev"] & split_ids["test"])
        assert "pair-0-a" in split_ids["train"] or "pair-0-a" in split_ids["dev"] or "pair-0-a" in split_ids["test"]
        for name in splits:
            assert ("pair-0-a" in split_ids[name]) == ("pair-0-b" in split_ids[name])
        all_test_ids.update(split_ids["test"])

    assert all_test_ids == {row["id"] for row in rows}
    assert folds == _split_kfold_train_dev_test(rows, folds=5, dev_portion=0.1, seed=42)
