import json
from datasets import Dataset,DatasetDict
from labs.lab03.src.data_preparation import export_data_review,make_tfidf


def test_eda_features_fit_train_only_and_keep_full_tokenization(tiny,tmp_path):
    _,tokenizer=tiny
    train=Dataset.from_dict({'text':['excellent movie','terrible film']*2,'label':[1,0]*2})
    held=Dataset.from_dict({'text':['heldoutuniquetoken one','heldoutuniquetoken two'],'label':[0,1]})
    raw=DatasetDict(train=train,validation=held,test=held)
    before=raw['validation']._fingerprint
    export_data_review(raw,tokenizer,{'max_length':16},tmp_path)
    features=json.loads((tmp_path/'feature_summary.json').read_text())
    assert features['fitted_split']=='train'
    assert not any('heldoutuniquetoken' in row['term'] for row in features['terms'])
    assert features['train_shape'][0]==4 and raw['validation']._fingerprint==before
    tokens=json.loads((tmp_path/'tokenization_examples.json').read_text())
    assert all(len(r['input_ids'])==len(r['attention_mask']) for r in tokens)
    assert len(list(tmp_path.glob('*.png')))==11
    quality=json.loads((tmp_path/'data_quality.json').read_text())
    assert quality['splits'][0]['normalized_duplicate_rows']==2
