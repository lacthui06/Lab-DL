import json
import numpy as np
from datasets import Dataset,DatasetDict
from labs.lab03.src.data_preparation import (
    export_data_review, make_tfidf, compute_summary_statistics,
    extract_linguistic_features, select_chi2_features
)


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
    assert len(list(tmp_path.glob('*.png')))==14
    quality=json.loads((tmp_path/'data_quality.json').read_text())
    assert quality['splits'][0]['normalized_duplicate_rows']==2
    prep_stats=json.loads((tmp_path/'preprocessing_statistics.json').read_text())
    assert 'summary_statistics' in prep_stats
    assert 'words' in prep_stats['summary_statistics']
    assert (tmp_path/'linguistic_features.json').exists()
    assert (tmp_path/'chi2_features.json').exists()


def test_compute_summary_statistics_iqr_outliers():
    values = [10, 12, 13, 14, 15, 16, 18, 100]  # 100 is an extreme outlier
    stats = compute_summary_statistics(values)
    assert stats['count'] == 8
    assert stats['outlier_count'] == 1
    assert stats['outlier_fraction'] == 0.125
    assert stats['median'] == 14.5


def test_extract_linguistic_features():
    texts = ["I LOVED this movie! It was great...", "No, terrible film."]
    feats = extract_linguistic_features(texts)
    assert len(feats) == 2
    assert feats[0]['uppercase_ratio'] > 0
    assert feats[0]['punctuation_ratio'] > 0
    assert feats[1]['negation_cue_count'] >= 1
