import json
from datasets import Dataset,DatasetDict
from src.training.trainer import make_trainer


def test_validation_plateau_stops_before_three_and_keeps_best_checkpoint(tiny,tmp_path):
    model,tokenizer=tiny
    data=Dataset.from_dict({**tokenizer(['excellent movie','terrible film']*2),'label':[1,0]*2})
    ds=DatasetDict(train=data,validation=data,test=data)
    cfg=dict(seed=42,num_train_epochs=3,batch_size=4,eval_batch_size=4,learning_rate=1e-5,
        weight_decay=.1,warmup_ratio=.1,fp16=False,device='cpu',eval_strategy='epoch',
        metric_for_best_model='loss',epoch_console_summary=True,save_only_model=True,record_resources=True,
        early_stopping={'enabled':True,'patience':1,'min_delta':.001})
    trainer=make_trainer(model,tokenizer,ds,cfg,tmp_path)
    original=trainer.evaluate
    def plateau(*args,**kwargs):
        if kwargs.get('metric_key_prefix','eval') != 'eval':return original(*args,**kwargs)
        metrics={'eval_loss':.8,'eval_accuracy':.5,'eval_macro_f1':.5}
        trainer.log(metrics)
        trainer.control=trainer.callback_handler.on_evaluate(trainer.args,trainer.state,trainer.control,metrics)
        return metrics
    trainer.evaluate=plateau
    try:trainer.train()
    finally:trainer.resource_callback.close()
    assert trainer.state.epoch==2
    assert trainer.state.best_model_checkpoint.endswith('checkpoint-1')
    assert [r['epoch'] for r in json.loads((tmp_path/'epoch_history.json').read_text())]==[1,2]
    assert len(json.loads((tmp_path/'resource_history.json').read_text()))==2


def test_active_profile_is_three_epochs_with_predeclared_regularization():
    from src.utils.runs import ROOT
    cfg=json.loads((ROOT/'config/frozen_trial.json').read_text())['exercise2']
    assert cfg['num_train_epochs']==3 and cfg['early_stopping']['enabled']
    assert cfg['early_stopping']['patience']==2 and cfg['metric_for_best_model']=='loss'
    assert cfg['augmentation']['enabled'] and cfg['weight_decay']>0
    assert cfg['model_overrides']['seq_classif_dropout']==.4
