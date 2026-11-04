"""EDA và feature engineering train-only dùng chung giữa notebook và training."""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .datasets.augmentation import normalized, PROTECTED
from .utils.runs import save_json
from .utils.plots import finish,plot_lengths
from .analysis import _table_image


def make_tfidf(settings):
    return TfidfVectorizer(ngram_range=tuple(settings['ngram_range']),
        min_df=settings['min_df'],max_features=settings['max_features'],sublinear_tf=True)


def export_data_review(raw, tokenizer, settings, output):
    """Heldout chỉ dùng thống kê chất lượng; token/features được fit/khảo sát trên train."""
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    quality=[];sets={}
    for split,ds in raw.items():
        texts=list(ds['text']);valid=[t for t in texts if isinstance(t,str)]
        normalized_texts=[normalized(t) for t in valid]
        quality.append({'split':split,'samples':len(ds),'missing_or_nonstring':len(texts)-len(valid),
            'empty_or_whitespace':sum(not t.strip() for t in valid),
            'exact_duplicate_rows':len(valid)-len(set(valid)),
            'normalized_duplicate_rows':len(valid)-len(set(normalized_texts)),
            'invalid_binary_labels':sum(y not in (0,1) for y in ds['label'])})
        sets[split]=set(normalized_texts)
    overlap={f'{a}__{b}':len(sets[a]&sets[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
    text=list(raw['train']['text']);labels=np.asarray(raw['train']['label'])
    if any(not isinstance(t,str) or not t.strip() for t in text):
        raise ValueError('Nonempty train strings required before feature extraction')
    ids=tokenizer(text,truncation=False)['input_ids']
    lengths=np.asarray([len(row) for row in ids]);words=np.asarray([len(t.split()) for t in text])
    statistics={'train_samples':len(text),'fitted_split':'train',
        'token_quantiles':{str(q):float(np.quantile(lengths,q)) for q in (.5,.9,.95,.99,1.)},
        'truncation_rates':{str(cut):float(np.mean(lengths>cut)) for cut in (64,96,128,256)},
        'configured_max_length':settings['max_length'],
        'preserve_negation_punctuation':True,'normalized_train_conflicting_label_groups':0}
    labelsets={}
    for t,y in zip(text,labels):labelsets.setdefault(normalized(t),set()).add(int(y))
    statistics['normalized_train_conflicting_label_groups']=sum(len(v)>1 for v in labelsets.values())
    signals=[]
    for label in (0,1):
        chosen=[t for t,y in zip(text,labels) if y==label]
        signals.append({'label':label,'samples':len(chosen),
            'mean_words':float(words[labels==label].mean()),'mean_tokens':float(lengths[labels==label].mean()),
            'negation_or_contrast_fraction':sum(bool(PROTECTED.search(t)) for t in chosen)/len(chosen),
            'digit_fraction':sum(bool(re.search(r'\d',t)) for t in chosen)/len(chosen)})
    save_json(output/'data_quality.json',{'splits':quality,'normalized_cross_split_overlap':overlap,
        'action':'audit only; official splits unchanged; no fitting with heldout text/labels'})
    save_json(output/'preprocessing_statistics.json',statistics)
    save_json(output/'text_signals.json',signals)
    _table_image(quality,['split','samples','missing_or_nonstring','empty_or_whitespace','normalized_duplicate_rows','invalid_binary_labels'],
        output/'data_quality.png','Data contract and quality counts; no automatic heldout-driven filtering',widths=[.15,.12,.2,.2,.2,.13])
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for label in (0,1):
        axes[0].hist(words[labels==label],bins=35,alpha=.55,label=f'Label {label}')
        axes[1].hist(lengths[labels==label],bins=35,alpha=.55,label=f'Label {label}')
    axes[0].set(xlabel='Whitespace words',ylabel='Train examples')
    axes[1].axvline(settings['max_length'],ls='--',color='black',label='Configured cutoff')
    axes[1].set(xlabel='WordPiece tokens incl. special tokens',ylabel='Train examples')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    finish(fig,output/'lengths_by_class.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].bar(list(statistics['truncation_rates']),[100*v for v in statistics['truncation_rates'].values()])
    axes[0].set(xlabel='Token cutoff',ylabel='Train truncated (%)',title='Cutoff sensitivity, train only')
    axes[1].bar(['Negative','Positive'],[100*r['negation_or_contrast_fraction'] for r in signals])
    axes[1].set(ylabel='Train reviews (%)',title='Negation/contrast: preserve linguistic cues')
    finish(fig,output/'preprocessing_diagnostics.png')
    indices=[]
    for label in (0,1):indices.extend(np.flatnonzero(labels==label)[:2].tolist())
    encoded=tokenizer([text[i] for i in indices],padding=True,truncation=True,max_length=settings['max_length'])
    examples=[]
    for j,index in enumerate(indices):
        tokens=tokenizer.convert_ids_to_tokens(encoded['input_ids'][j])
        examples.append({'index':index,'text':text[index],'label':int(labels[index]),
            'tokens':tokens,'input_ids':encoded['input_ids'][j],'attention_mask':encoded['attention_mask'][j],
            'tokens_preview':' '.join(tokens[:12]),'ids_preview':' '.join(map(str,encoded['input_ids'][j][:12]))})
    save_json(output/'tokenization_examples.json',examples)
    basic=[{k:r[k] for k in ['index','text','label']} for r in examples]
    counts={split:{str(c):sum(int(y)==c for y in ds['label']) for c in (0,1)} for split,ds in raw.items()}
    save_json(output/'data_preview.json',{'split_sizes':{k:len(v) for k,v in raw.items()},
        'label_counts':counts,'train_token_lengths':{'mean':float(lengths.mean()),'max':int(lengths.max()),
        'truncated_fraction':float(np.mean(lengths>settings['max_length']))},
        'examples':basic,'examples_split':'train','augmentation_applied':False})
    plot_lengths(lengths,settings['max_length'],output/'token_length_distribution.png')
    display_rows=[{**r,'text':r['text'][:130]+('...' if len(r['text'])>130 else '')} for r in examples]
    _table_image(display_rows,['index','text','label'],output/'train_examples.png','Original train examples (text preview; full text in JSON)',widths=[.08,.84,.08])
    fig,axes=plt.subplots(1,len(counts),figsize=(5*len(counts),4),squeeze=False)
    for ax,(split,support) in zip(axes[0],counts.items()):
        bars=ax.bar(['Negative','Positive'],[support['0'],support['1']]);ax.bar_label(bars)
        ax.set(title=split,ylabel='Official examples')
    finish(fig,output/'class_distribution.png')
    _table_image(display_rows,['text','tokens_preview','ids_preview'],output/'preprocessing_examples.png',
        'Actual train tokenization: first 12 tokens/IDs; full tensors in JSON',widths=[.4,.35,.25])
    fig,ax=plt.subplots(figsize=(12,3))
    image=ax.imshow(encoded['attention_mask'],aspect='auto',vmin=0,vmax=1,cmap='Blues')
    ax.set(xlabel='Position after dynamic padding',ylabel='Train example',yticks=range(len(indices)),yticklabels=indices)
    fig.colorbar(image,ax=ax,ticks=[0,1],label='0=padding, 1=valid token')
    finish(fig,output/'attention_mask.png')
    feature_settings={'ngram_range':[1,2],'min_df':2,'max_features':50000}
    vectorizer=make_tfidf(feature_settings);matrix=vectorizer.fit_transform(text)
    names=vectorizer.get_feature_names_out();terms=[]
    for label in (0,1):
        means=np.asarray(matrix[labels==label].mean(axis=0)).ravel()
        for index in np.argsort(means)[-12:][::-1]:
            terms.append({'label':label,'term':str(names[index]),'mean_tfidf':float(means[index])})
    features={'fitted_split':'train','settings':feature_settings,'train_shape':list(matrix.shape),'nnz':int(matrix.nnz),
        'density':float(matrix.nnz/(matrix.shape[0]*matrix.shape[1])),
        'sparse_array_storage_bytes':int(matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes),
        'storage_note':'array footprint only, not process RAM peak','terms':terms}
    save_json(output/'feature_summary.json',features)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for label,ax in enumerate(axes):
        group=[r for r in terms if r['label']==label][::-1]
        ax.barh([r['term'] for r in group],[r['mean_tfidf'] for r in group])
        ax.set(title=f'Train mean TF-IDF, label {label}',xlabel='Mean feature value')
    finish(fig,output/'tfidf_features.png')
    return statistics,features
