"""Synthetic examples chỉ thuộc train, tái lập được và không đổi heldout."""
import json
import pytest
from datasets import Dataset, DatasetDict
from labs.lab03.src.datasets.augmentation import augment_train, normalized, export_augmentation


def fixture():
    train=Dataset.from_dict({'text':['a wonderful story one','an excellent story two','a charming story three','an enjoyable story four',
        'a terrible story one','a boring story two','a dull story three','an awful story four'],'label':[1]*4+[0]*4})
    return DatasetDict(train=train,validation=Dataset.from_dict({'text':['this is heldout validation']}),
        test=Dataset.from_dict({'text':['this is heldout test']}))


def test_augmented_train_has_new_balanced_examples_and_heldout_is_unchanged():
    raw=fixture();old={k:raw[k].to_dict() for k in raw}
    result,summary=augment_train(raw,{'enabled':True,'max_ratio':.5},42)
    assert summary['added_count']==4 and len(result['train'])==12
    assert summary['added_per_class']=={'0':2,'1':2}
    assert result['train'].select(range(8)).to_dict()==old['train']
    for split in ['validation','test']:
        assert result[split].to_dict()==old[split]
        assert result[split]._fingerprint==raw[split]._fingerprint
    texts=list(map(normalized,result['train']['text']));assert len(texts)==len(set(texts))
    for r in summary['records']:assert r['label']==raw['train'][r['parent_index']]['label']
    repeat,again=augment_train(raw,{'enabled':True,'max_ratio':.5},42)
    assert summary==again and repeat['train'].to_dict()==result['train'].to_dict()


def test_negation_contrast_and_heldout_collisions_are_rejected():
    raw=fixture();raw['train']=Dataset.from_dict({'text':['not a wonderful story','a boring story but good acting',
        'a wonderful story one','a delightful story two','a terrible story one','a dull story two'],'label':[1,0,1,1,0,0]})
    raw['test']=Dataset.from_dict({'text':['a excellent story one']})
    result,summary=augment_train(raw,{'enabled':True,'max_ratio':.5},42)
    assert not {0,1}&{r['parent_index'] for r in summary['records']}
    assert summary['heldout_text_collisions_rejected']>=1
    assert not set(map(normalized,result['train']['text']))&set(map(normalized,raw['test']['text']))


@pytest.mark.parametrize('ratio',[0,-1,1,float('nan'),True])
def test_augmentation_rejects_invalid_ratio(ratio):
    with pytest.raises(ValueError):augment_train(fixture(),{'enabled':True,'max_ratio':ratio})


def test_augmentation_exports_records_and_real_images(tmp_path):
    _,summary=augment_train(fixture(),{'enabled':True,'max_ratio':.5})
    export_augmentation(summary,tmp_path)
    assert json.loads((tmp_path/'augmentation.json').read_text())['records']==summary['records']
    assert (tmp_path/'augmentation_counts.png').is_file() and (tmp_path/'augmentation_examples.png').is_file()


def test_finetune_uses_added_train_examples_but_keeps_official_split_contract(tiny,tmp_path,monkeypatch):
    from labs.lab03.src import workflow
    from labs.lab03.src.datasets.loader import tokenize_data
    from labs.lab03.src.utils.runs import ROOT,run_context
    model,tok=tiny;initial=tmp_path/'initial';model.save_pretrained(initial);tok.save_pretrained(initial)
    raw=fixture()
    for split in ['validation','test']:
        raw[split]=Dataset.from_dict({'text':['excellent movie','terrible film'],'label':[1,0]})
    monkeypatch.setattr(workflow,'prepare_data',lambda config:(raw,tok,tokenize_data(raw,tok,16)))
    monkeypatch.setattr(workflow,'run_context',lambda kind,cfg:run_context(kind,cfg,root=tmp_path))
    cfg=workflow.read_config(ROOT/'config/frozen_trial.json')
    cfg['exercise2'].update(model_name=str(initial),model_revision=None,batch_size=4,eval_batch_size=4,fp16=False)
    cfg['exercise2']['augmentation']['max_ratio']=.5
    out=workflow.finetune(cfg,allow_cpu=True,evaluate_test=False)
    stats=json.loads((out/'data_statistics.json').read_text())
    assert stats['split_sizes']=={'train':8,'validation':2,'test':2}
    assert stats['effective_split_sizes']=={'train':12,'validation':2,'test':2}
    assert json.loads((out/'selection_summary.json').read_text())['global_step']<=15
    assert 1 <= len(json.loads((out/'epoch_history.json').read_text())) <= 5
    assert json.loads((out/'augmentation.json').read_text())['added_count']==4
    assert not (out/'all_predictions.json').exists()
