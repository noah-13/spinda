import io
import json
import math
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

from spinda.data import TextPairClassificationJSONReader, SingleTextMultilabelJSONReader, SingleTextMultilevelJSONReader
from spinda.data.schemas import PredictionRecord
from spinda.eval import Evaluator
from spinda.models.trainer import HLVTrainer, TrainingConfig, SoftLabelTrainer
from spinda.scripts.evaluate import load_human_labels, _infer_prediction_kind
from spinda.scripts.predict import predict_batch
from spinda.scripts.prepare_discogem_annotation_labels import _rows, _votes, LABELS
from spinda.scripts.prepare_humans_and_domains_annotation_labels import prepare_humans_and_domains


def test_categorical_mse_sums_classes_and_averages_examples():
    logits = torch.zeros((2, 3), requires_grad=True)
    targets = torch.tensor([[1., 0., 0.], [1/3, 1/3, 1/3]])
    loss = SoftLabelTrainer._compute_soft_loss(SimpleNamespace(soft_label_loss="mse"), logits, targets)
    assert loss.item() == pytest.approx(1/3)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("multilabel", [False, True])
def test_training_jsd_uses_bits(multilabel):
    logits = torch.zeros((1, 2), requires_grad=True)
    target = torch.tensor([[1., 0.]])
    if multilabel:
        trainer = SimpleNamespace(head_type="multilabel_classification", soft_label_loss="jsd")
        loss = SoftLabelTrainer.compute_loss(trainer, lambda **kw: SimpleNamespace(logits=logits), {"labels": target})
    else:
        loss = SoftLabelTrainer._compute_soft_loss(SimpleNamespace(soft_label_loss="jsd"), logits, target)
    # JSD([1,0], [.5,.5]) in bits.
    expected = .5 * math.log2(4/3) + .25 * math.log2(2/3) + .25
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_discogem_reads_absolute_counts_and_preserves_level_alignment(tmp_path):
    item = "item"
    header = "itemid\tsplit\targ1_context_en\targ2_context_en\tMV_dist_en\n"
    items = header + "item\ttrain\ta\tb\t{'reason': 0.6, 'contrast': 0.4}\n"
    labels = "task\tworker\tlabel\n" + "".join(f"item.en\tw{i}\t{label}\n" for i, label in enumerate(["reason", "contrast"] * 4 + ["reason", "reason", "norel"]))
    archive = tmp_path / "data.tgz"
    with tarfile.open(archive, "w:gz") as handle:
        for name, data in [("items", items), ("labels", labels)]:
            blob = data.encode(); info = tarfile.TarInfo(f"DiscoGeM2.0_annotation/DiscoGeM2.0_{name}.csv"); info.size = len(blob)
            handle.addfile(info, io.BytesIO(blob))
    votes = _rows(archive)[0]["annotation_votes_en"]
    assert len(votes) == 10
    assert _votes(votes, "level3").count(LABELS["level3"].index("reason")) == 6
    assert _votes(votes, "level1") == [1, 2] * 4 + [1, 1]


@pytest.mark.parametrize("kind,annotations,outputs", [
    ("categorical", {"annotation_labels": [0, 1]}, {"probs": [.4,.4,.2], "pred": 0}),
    ("multilabel", {"annotation_label_sets": [[0], []]}, {"probs": [.5,.2,.1], "pred": [1,0,0]}),
    ("multilevel", {"annotation_labels": {"genre": [0,1]}}, {"dimensions": {"genre": {"probs": [.4,.4,.2], "pred": 0}}}),
])
def test_evaluation_retains_classes_absent_from_annotations(tmp_path, kind, annotations, outputs):
    path = tmp_path / "test.json"; path.write_text(json.dumps([{"id":"x", "text":"hi", **annotations}]))
    predictions = [PredictionRecord(id="x", task="test", outputs=outputs)]
    gold = load_human_labels(path, kind, predictions)[0]
    actual = gold.human_dists["genre"] if kind == "multilevel" else gold.human_probs if kind == "multilabel" else gold.human_dist
    assert len(actual) == 3 and actual[-1] == 0
    bad = [PredictionRecord(id="x", task="test", outputs={"probs":[1.],"pred":0})]
    if kind == "categorical":
        with pytest.raises(ValueError, match="label space"):
            load_human_labels(path, kind, bad)


def test_rejects_legacy_levels_field():
    with pytest.raises(ValueError, match="no recognized output shape"):
        _infer_prediction_kind([PredictionRecord(id="x", task="test", outputs={"levels":{"genre":{"probs":[.4,.6],"pred":1}}})])


def test_opt_in_metrics_do_not_compute_distance_correlation(tmp_path, monkeypatch):
    path=tmp_path / "test.json";path.write_text(json.dumps([{"id":"x","text":"hi","annotation_labels":[0,1,1]}]))
    predictions=[PredictionRecord(id="x",task="test",outputs={"probs":[.3,.7],"pred":1})]
    import spinda.eval.evaluator as module
    monkeypatch.setattr(module, "compute_distance_correlation", lambda *a: pytest.fail("unrequested expensive metric"))
    result=Evaluator(distribution_metrics=["macro_f1","soft_accuracy","tvd"]).evaluate(predictions,load_human_labels(path,"categorical",predictions)).metrics
    assert result["macro_f1"] == .5
    assert result["soft_accuracy"] == pytest.approx(1-result["tvd"])


def test_tgegum_normalizes_topic2_and_matches_separate_joint_ties(tmp_path):
    raw=tmp_path/'raw';raw.mkdir()
    for split in ['train','dev','test']:
        (raw/f'{split}-sent.json').write_text(json.dumps({'x':{'text':'hi','annotations-genre':['a','b','c'],'annotations-topic1':['No Topic','a','b'],'annotations-topic2':['No topic','No Topic','other']}}))
    prepare_humans_and_domains(raw,tmp_path/'prepared')
    root=tmp_path/'prepared/humans_and_domains'
    config=json.loads((root/'topic2/dataset.json').read_text())
    assert config['labels']==['No Topic','other']
    from spinda.data import SingleTextClassificationJSONReader
    joint=SingleTextMultilevelJSONReader(str(root/'multilevel')).load_test()[0]
    for level,task in zip(['level1','level2','level3'],['genre','topic1','topic2']):
        separate=SingleTextClassificationJSONReader(str(root/task)).load_test()[0]
        assert separate.label == joint.hard_labels[level]
        assert load_human_labels(root/task/'test.json','categorical')[0].label == separate.label
    assert load_human_labels(root/'multilevel/test.json','multilevel')[0].hard_labels == joint.hard_labels


@pytest.fixture(scope='module')
def tiny_backbone(tmp_path_factory):
    previous=torch.get_num_threads();torch.set_num_threads(1)
    p=tmp_path_factory.mktemp('tiny_backbone')
    (p/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n')
    BertTokenizerFast(vocab_file=str(p/'vocab.txt'),model_max_length=32).save_pretrained(p)
    BertModel(BertConfig(vocab_size=7,hidden_size=8,num_hidden_layers=1,num_attention_heads=2,intermediate_size=16,max_position_embeddings=32)).save_pretrained(p)
    yield p
    torch.set_num_threads(previous)


def test_seed_controls_random_head_initialization(tiny_backbone):
    weights=[]
    for seed in [42,42,43]:
        trainer=HLVTrainer(TrainingConfig(model_name_or_path=str(tiny_backbone),num_labels=2,seed=seed))
        trainer.initialize_model();weights.append(trainer.model.classifier.weight.detach().clone())
    assert torch.equal(weights[0],weights[1])
    assert not torch.equal(weights[0],weights[2])


@pytest.mark.parametrize('kind',['pair','multilabel','dimensions'])
@pytest.mark.parametrize('mode,strategy',[('soft','mse'),('soft','jsd'),('soft','rel'),('soft_to_hard','ce')])
def test_offline_train_save_predict_evaluate(tiny_backbone,tmp_path,kind,mode,strategy):
    formats={'pair':'text_pair_label_distribution','multilabel':'single_text_multilabel_annotation_distribution','dimensions':'single_text_multidimensional_label_distribution'}
    rows=[]
    for i,votes in enumerate([[0,0],[0,1],[1,1]]):
        row={'id':f'x{i}','text':'hello world'}
        if kind=='pair': row={'id':f'x{i}','text_a':'hello','text_b':'world','annotation_labels':votes}
        elif kind=='multilabel':row['annotation_label_sets']=[[v] for v in votes]
        else:row['annotation_labels']={'genre':votes,'topic':list(reversed(votes))}
        rows.append(row)
    manifest={'format':formats[kind],'label_mode':mode}
    manifest['level_labels' if kind=='dimensions' else 'labels']={'genre':['a','b'],'topic':['a','b']} if kind=='dimensions' else ['a','b']
    (tmp_path/'dataset.json').write_text(json.dumps(manifest))
    for split in ['train','dev','test']:(tmp_path/f'{split}.json').write_text(json.dumps(rows))
    reader={'pair':TextPairClassificationJSONReader,'multilabel':SingleTextMultilabelJSONReader,'dimensions':SingleTextMultilevelJSONReader}[kind](str(tmp_path))
    config=TrainingConfig(model_name_or_path=str(tiny_backbone),num_labels=2,num_epochs=1,train_batch_size=3,eval_batch_size=3,max_length=16,device='cpu',dataloader_num_workers=0,output_dir=str(tmp_path/'run'),use_soft_labels=mode=='soft',use_soft_eval_metrics=mode=='soft_to_hard',soft_label_loss=strategy,head_type={'pair':'classification','multilabel':'multilabel_classification','dimensions':'multilevel_classification'}[kind],multilevel_label_sizes={'genre':2,'topic':2} if kind=='dimensions' else None)
    trainer=HLVTrainer(config);trainer.train(reader.load_train(),reader.load_dev())
    final=tmp_path/'run/final_model'
    assert json.loads((final/'training_config.json').read_text())['max_length']==16
    restored=HLVTrainer(config);restored.load_model(str(final))
    outputs=predict_batch(restored.model,restored.tokenizer,['hello world']*3,None,device='cpu',max_length=16)
    predictions=[PredictionRecord(id=r['id'],task='test',outputs=o) for r,o in zip(rows,outputs)]
    ground_truth=load_human_labels(tmp_path/'test.json',{'pair':'categorical','multilabel':'multilabel','dimensions':'multilevel'}[kind],predictions)
    if kind=='dimensions':
        from spinda.scripts.evaluate import _build_multilevel_eval_inputs
        for level in ['genre','topic']:
            pp,gg=_build_multilevel_eval_inputs(predictions,ground_truth,level)
            assert np.isfinite(list(Evaluator().evaluate(pp,gg).metrics.values())).all()
    else:
        assert np.isfinite(list(Evaluator().evaluate(predictions,ground_truth).metrics.values())).all()
    checkpoints=list((tmp_path/'run').glob('checkpoint-*/trainer_state.json'))
    assert checkpoints and json.loads(checkpoints[0].read_text())['best_metric'] is not None


def test_cli_pipeline_uses_saved_length_and_exposes_requested_metrics(tiny_backbone, tmp_path, monkeypatch):
    import sys
    from spinda.scripts import train, predict, evaluate
    records=[{'id':str(i),'text':'hello world','annotation_labels':v} for i,v in enumerate([[0,0],[0,1],[1,1]])]
    data=tmp_path/'data.json';data.write_text(json.dumps(records))
    config=tmp_path/'dataset.json';config.write_text(json.dumps({'format':'single_text_label_distribution','labels':['a','b'],'label_mode':'soft','train_path':str(data),'dev_path':str(data)}))
    output=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['train','--config',str(config),'--model',str(tiny_backbone),'--device','cpu','--num_epochs','1','--max_length','12','--dataloader_num_workers','0','--output_dir',str(output)])
    train.main()
    run_config=json.loads((output/'seed_42/run_config.json').read_text())
    assert run_config['seeds']==[42] and len(run_config['data_sha256']['train'])==64
    seen=[];original=predict.predict_batch
    def capture(*args, **kwargs):
        seen.append(kwargs['max_length']);return original(*args,**kwargs)
    monkeypatch.setattr(predict,'predict_batch',capture)
    predictions=tmp_path/'predictions.json'
    monkeypatch.setattr(sys,'argv',['predict','--model_path',str(output/'seed_42/final_model'),'--input_file',str(data),'--output_file',str(predictions),'--device','cpu'])
    predict.main()
    assert seen==[12]
    result=tmp_path/'evaluation.json'
    monkeypatch.setattr(sys,'argv',['evaluate','--predictions',str(predictions),'--input_file',str(data),'--human_labels',str(data),'--metrics','macro_f1','soft_accuracy','--output_file',str(result)])
    evaluate.main()
    assert set(json.loads(result.read_text()))=={'macro_f1','soft_accuracy'}


@pytest.mark.parametrize('plots',[['stratified'],['instance'],['stratified','instance']])
def test_analyze_handles_empty_strata(tmp_path,monkeypatch,plots):
    import sys
    from spinda.scripts import analyze
    path=tmp_path/'analysis.json';path.write_text(json.dumps({'disagreement_stratified':{'thresholds':{'boundaries':[0,0]},'groups':{'low':{'n':2,'metrics':{'tvd':.2}},'medium':{'n':0,'metrics':None},'high':{'n':0,'metrics':None}}}}))
    errors=tmp_path/'errors.csv';errors.write_text('id,tvd\na,0.1\nb,0.3\n')
    monkeypatch.setattr(sys,'argv',['analyze','--analysis-files',str(path),'--labels','model','--instance-errors-files',str(errors),'--plots',*plots,'--output-dir',str(tmp_path/'plots')])
    analyze.main()
    if 'stratified' in plots:assert (tmp_path/'plots/disagreement_tvd.png').is_file()
    if 'instance' in plots:assert (tmp_path/'plots/instance_tvd_violin.png').is_file()


def test_analyze_rejects_different_thresholds(tmp_path,monkeypatch):
    import sys
    from spinda.scripts import analyze
    paths=[]
    for i in range(2):
        path=tmp_path/f'{i}.json';path.write_text(json.dumps({'disagreement_stratified':{'thresholds':{'boundaries':[.1+i*.1,.5]},'groups':{'low':{'n':1,'metrics':{'tvd':.2}},'medium':{'n':0,'metrics':None},'high':{'n':0,'metrics':None}}}}));paths.append(str(path))
    monkeypatch.setattr(sys,'argv',['analyze','--analysis-files',*paths,'--labels','a','b','--plots','stratified'])
    with pytest.raises(ValueError,match='boundaries'):analyze.main()
