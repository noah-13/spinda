import json

from spinda.data import SingleTextMultilabelJSONReader
from spinda.scripts.prepare_mfrc_annotation_labels import prepare_mfrc


def test_prepare_mfrc_groups_annotators_and_reader_preserves_votes(tmp_path):
    rows = [
        {"text": "A comment", "subreddit": "politics", "bucket": "US Politics", "annotator": "a", "annotation": "Care,Equality", "confidence": "Confident"},
        {"text": "A comment", "subreddit": "politics", "bucket": "US Politics", "annotator": "b", "annotation": "Care", "confidence": "Somewhat Confident"},
        {"text": "A comment", "subreddit": "politics", "bucket": "US Politics", "annotator": "c", "annotation": "Non-Moral", "confidence": "Confident"},
    ]
    output = tmp_path / "mfrc"
    counts = prepare_mfrc(rows, output)
    assert sum(counts.values()) == 1
    manifest = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "single_text_multilabel_annotation_distribution"
    assert manifest["source"]["dataset"] == "USC-MOLA-Lab/MFRC"
    assert manifest["source"]["split"] == "train_dedup"
    split = next(split for split, count in counts.items() if count)
    sample = SingleTextMultilabelJSONReader(str(output)).load_split(split)[0]
    assert sample.annotation_label_sets == [[0, 1], [0], [7]]
    assert sample.human_probs == [2 / 3, 1 / 3, 0.0, 0.0, 0.0, 0.0, 0.0, 1 / 3]
    assert sample.labels == [1, 0, 0, 0, 0, 0, 0, 0]
