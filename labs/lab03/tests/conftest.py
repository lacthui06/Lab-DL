import pytest
import torch
from transformers import DistilBertTokenizerFast, DistilBertConfig, DistilBertForSequenceClassification

@pytest.fixture
def tiny(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "i", "loved", "this", "excellent", "movie", "a", "terrible", "boring", "film"]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocab))
    tokenizer = DistilBertTokenizerFast(vocab_file=str(path), do_lower_case=True)
    # DistilBERT không nhận token_type_ids.
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    cfg = DistilBertConfig(vocab_size=len(vocab), n_layers=1, dim=32, hidden_dim=64,
                          n_heads=2, dropout=0, attention_dropout=0, seq_classif_dropout=0,
                          num_labels=2, id2label={0: "NEGATIVE", 1: "POSITIVE"},
                          label2id={"NEGATIVE": 0, "POSITIVE": 1})
    return DistilBertForSequenceClassification(cfg), tokenizer


@pytest.fixture
def audit_trial(tmp_path):
    """Synthetic contract fixture in pytest tmp only; never training/report evidence."""
    import hashlib
    import json
    import numpy as np
    from sklearn.metrics import log_loss
    from labs.lab03.src.utils.runs import ROOT, save_json, hash_file
    from labs.lab03.src.training.thresholds import binary_metrics_at_threshold, select_validation_threshold
    from labs.lab03.src.utils.plots import plot_epoch_learning
    folder = tmp_path/'synthetic_trial_fixture'
    folder.mkdir()
    cfg = json.loads((ROOT/'config/frozen_trial.json').read_text())
    cfg.update(fixture_only=True, session_id='synthetic-test-session')
    labels = np.array([0]*533+[1]*533)
    def records(period, low, high):
        positive = np.where(labels==1, high, low)
        positive[::period] = 1-positive[::period]
        return [{'index':i,'text':f'Synthetic contract fixture {i}',
            'true_label':int(y),'probabilities':[float(1-p),float(p)]}
            for i,(y,p) in enumerate(zip(labels,positive))]
    validation, test, baseline = records(7,.1,.9), records(5,.2,.8), records(4,.3,.7)
    policy = select_validation_threshold(labels, [r['probabilities'][1] for r in validation])
    loss = float(log_loss(labels,[r['probabilities'] for r in validation]))
    rows = [{'epoch':i,'step':i,'train_loss_eval':loss+.1/i,'train_accuracy':.8,
        'validation_loss':loss+.1*(3-i),'validation_accuracy':.8,
        'validation_macro_f1':.8,'epoch_runtime_seconds':1.} for i in range(1,4)]
    weights_hash = hashlib.sha256(b'synthetic placeholder; no model weights').hexdigest()
    objects = {'config.json':cfg, 'status.json':{'status':'completed','fixture_only':True},
        'environment_snapshot.json':{'device':'cpu','cuda_available':False,'fixture_only':True},
        'epoch_history.json':rows,
        'selection_summary.json':{'completed_epochs':3,'requested_epochs':3,'early_stopping':True,
            'selected_epoch':3,'best_step':3,'test_used_for_selection':False},
        'frozen_selection.json':{'selected_epoch':3,'threshold':policy['selected_threshold'],
            'weights_sha256':weights_hash,'test_used_for_selection':False},
        'data_statistics.json':{'official_sizes':{'train':8530,'validation':1066,'test':1066},
            'effective_sizes':{'train':8530,'validation':1066,'test':1066},'validation_and_test_augmented':False},
        'augmentation.json':{'added_count':0,'records':[],
            'records_sha256':hashlib.sha256(b'[]').hexdigest(),'validation_and_test_modified':False},
        'freezing_and_optimizer.json':{'frozen_embeddings':True,'frozen_encoder_layers':cfg['exercise2']['freeze_layers'],
            'optimizer_groups':[{'parameter_names':['classifier.weight'],'lr':cfg['exercise2']['head_learning_rate']}]},
        'validation_predictions.json':validation,'test_predictions.json':test,
        'baseline_test_predictions.json':baseline,'decision_threshold.json':policy,
        'test_runtime_metrics.json':{'test_loss':float(log_loss(labels,[r['probabilities'] for r in test])),
            'test_runtime':1.,'fixture_only':True}}
    comparison=[]
    for name,data,threshold in [('TF-IDF',baseline,.5),('Frozen DistilBERT threshold 0.5',test,.5),
            ('Frozen DistilBERT validation threshold',test,policy['selected_threshold'])]:
        comparison.append({'model':name,**binary_metrics_at_threshold(labels,[r['probabilities'][1] for r in data],threshold)})
    objects['test_comparison.json']=comparison
    # Values are synthetic telemetry inputs for unit conversion tests, not hardware measurements.
    objects['resource_history.json']=[{'epoch':i,'ms_per_sample':.1,'ram_peak_rss_mib':128.,
        'ram_mib_per_batch_slot':8.,'vram_peak_allocated_mib':None,'vram_peak_reserved_mib':None,
        'vram_mib_per_batch_slot':None,'device':'cpu','scope':'synthetic test inputs'} for i in range(1,4)]
    for name,value in objects.items():save_json(folder/name,value)
    (folder/'console.log').write_text('SYNTHETIC PYTEST FIXTURE. Not a training run.\n')
    history=[]
    for row in rows:
        history += [{'epoch':row['epoch'],'step':row['step'],'train_loss':row['train_loss_eval'],'train_accuracy':.8},
            {'epoch':row['epoch'],'step':row['step'],'eval_loss':row['validation_loss'],'eval_accuracy':.8}]
    plot_epoch_learning(history,folder/'learning_curves.png')
    import shutil
    shutil.copyfile(folder/'learning_curves.png',folder/'validation_threshold_curves.png')
    from labs.lab03.src.performance import write_performance
    write_performance(folder,folder)
    save_json(folder/'manifest.json',{**{p.name:hash_file(p) for p in folder.iterdir()},
        'best_model/model.safetensors':weights_hash})
    return folder


@pytest.fixture
def exercise_run(tmp_path):
    """Inference contract fixture; pipeline/manual values are synthetic test inputs."""
    from labs.lab03.src.utils.runs import ROOT, save_json, hash_file
    import json
    folder=tmp_path/'synthetic_exercise_fixture';folder.mkdir()
    cfg=json.loads((ROOT/'config/frozen_trial.json').read_text());cfg['fixture_only']=True
    manual=[{'text':text,'label':'POSITIVE','score':.8,'probabilities':[.2,.8]}
            for text in cfg['exercise1']['sample_sentences']]
    pipeline=[{k:row[k] for k in ('label','score')} for row in manual]
    save_json(folder/'config.json',cfg)
    save_json(folder/'status.json',{'status':'completed','fixture_only':True})
    save_json(folder/'predictions.json',{'manual':manual,'pipeline':pipeline})
    save_json(folder/'tokenizer_breakdown.json',{'text':'Synthetic tokenizer contract fixture','input_ids':[101,1,2,3,4,5,6,102],
        'tokens':['[CLS]','synthetic','unit','test','tokens','for','contract','[SEP]'],
        'attention_mask':[1]*8})
    save_json(folder/'manifest.json',{p.name:hash_file(p) for p in folder.iterdir()})
    return folder
