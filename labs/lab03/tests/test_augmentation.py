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
    cfg=workflow.read_config(ROOT/'config/augmented.json')
    cfg['exercise2'].update(model_name=str(initial),model_revision=None,batch_size=4,eval_batch_size=4,fp16=False)
    cfg['exercise2']['augmentation']['max_ratio']=.5
    out=workflow.finetune(cfg,allow_cpu=True,evaluate_test=False)
    stats=json.loads((out/'data_statistics.json').read_text())
    assert stats['split_sizes']=={'train':8,'validation':2,'test':2}
    assert stats['effective_split_sizes']=={'train':12,'validation':2,'test':2}
    assert json.loads((out/'selection_summary.json').read_text())['global_step']==30
    assert len(json.loads((out/'epoch_history.json').read_text()))==10
    assert json.loads((out/'augmentation.json').read_text())['added_count']==4
    assert not (out/'all_predictions.json').exists()


def test_augmented_study_uses_identical_pool_for_baseline_and_model_and_passes_audit(tiny,tmp_path,monkeypatch):
    from labs.lab03.src import workflow,study
    from labs.lab03.src.study_evidence import audit_study
    from labs.lab03.src.utils.runs import ROOT,run_context
    from labs.lab03.src.datasets.loader import tokenize_data
    model,tok=tiny;initial=tmp_path/'initial';model.save_pretrained(initial);tok.save_pretrained(initial)
    raw=fixture()
    for split in ['validation','test']:
        raw[split]=Dataset.from_dict({'text':['excellent movie','terrible film'],'label':[1,0]})
    monkeypatch.setattr(study,'ROOT',tmp_path)
    monkeypatch.setattr(study,'require_study_gpu',lambda:None)
    monkeypatch.setattr(study,'load_data',lambda cfg:raw)
    monkeypatch.setattr(study,'run_context',lambda kind,cfg:run_context(kind,cfg,root=tmp_path))
    monkeypatch.setattr(workflow,'prepare_data',lambda cfg:(raw,tok,tokenize_data(raw,tok,16)))
    monkeypatch.setattr(workflow,'run_context',lambda kind,cfg:run_context(kind,cfg,root=tmp_path))
    original=workflow.finetune
    monkeypatch.setattr(study,'finetune',lambda cfg,evaluate_test:original(cfg,allow_cpu=True,evaluate_test=evaluate_test))
    cfg=workflow.read_config(ROOT/'config/augmented.json')
    cfg['exercise2'].update(model_name=str(initial),model_revision=None,batch_size=4,eval_batch_size=4,fp16=False)
    cfg['exercise2']['augmentation']['max_ratio']=.5
    settings={'seed':42,'epochs_per_candidate':10,'learning_rates':[1e-5],
        'baseline':{'C_values':[1],'ngram_range':[1,2],'min_df':1,'max_features':50},'bootstrap_repeats':10}
    out=study.run_study(cfg,settings)
    audited=audit_study(out,root=tmp_path,expected_sizes={'train':8,'validation':2,'test':2})
    assert audited['status']=='passed'
    assert (out/'augmentation_counts.png').is_file()
    aug=json.loads((out/'augmentation.json').read_text());assert aug['effective_train_size']==12
    with run_context('augmentation_preview',cfg,root=tmp_path) as preview:
        export_augmentation(aug,preview)
    import importlib.util
    spec=importlib.util.spec_from_file_location('augmented_package',ROOT/'scripts/package_study.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    review,_=module.package_study(out,include_checkpoint=False,root=tmp_path)
    from zipfile import ZipFile
    with ZipFile(review) as archive:
        assert f'outputs/runs/{preview.name}/augmentation.json' in archive.namelist()
