from nli_toolkits.data.schemas import (
    NLISample, NLIDistributionSample,
    NLI_LABELS, NLI_LABEL2ID, NLI_ID2LABEL, NLI_NUM_LABELS,
    PredictionRecord,
)

def test_labels_meta():
    assert NLI_LABELS == ["entailment", "neutral", "contradiction"]
    assert NLI_NUM_LABELS == 3
    assert NLI_LABEL2ID["neutral"] == 1
    assert NLI_ID2LABEL[0] == "entailment"

def test_nli_sample_construct():
    s = NLISample(
        id="ex1", task="nli", split="train", source="snli",
        premise="A cat sits on the mat.",
        hypothesis="An animal is sitting.",
        label=0,
    )
    
    assert s.label == 0
    assert s.premise.startswith("A cat")

def test_nli_distribution_sample_and_prediction():
    d = NLIDistributionSample(
        id="ex2", task="nli", split="test", source="chaosnli-snli",
        premise="A dog runs.",
        hypothesis="An animal is moving.",
        label=0,
        human_dist=[0.6, 0.3, 0.1],
    )
    assert len(d.human_dist) == NLI_NUM_LABELS
    assert abs(sum(d.human_dist) - 1.0) < 1e-6

    pred = PredictionRecord(
        id="ex1", task="nli", split="test", source="snli",
        outputs={"probs": [0.7, 0.2, 0.1], "pred": 0},
    )
    assert pred.outputs["pred"] == 0