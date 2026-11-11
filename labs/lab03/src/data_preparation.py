"""EDA và feature engineering train-only dùng chung giữa notebook và training."""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .datasets.augmentation import normalized, PROTECTED
from .utils.runs import save_json
from .utils.plots import finish,plot_lengths
from .analysis import _table_image


NEGATION_CONTRAST = PROTECTED
NEGATION_CUES = re.compile(r"\b(?:not|no|never|neither|nor|hardly|barely|without)\b|n['’]t\b", re.I)
CONTRAST_CUES = re.compile(r"\b(?:but|however|although|despite|yet)\b", re.I)


def compute_summary_statistics(values):
    """Tính 5-number summary (Min, Q1, Median, Q3, Max), Mean, Std và IQR Outliers theo docs/eda.md."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {}
    q1 = float(np.quantile(arr, 0.25))
    median = float(np.median(arr))
    q3 = float(np.quantile(arr, 0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = arr[(arr < lower_bound) | (arr > upper_bound)]
    return {
        'count': int(len(arr)),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'q1': q1,
        'median': median,
        'q3': q3,
        'max': float(np.max(arr)),
        'iqr': float(iqr),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'outlier_count': int(len(outliers)),
        'outlier_fraction': float(len(outliers) / len(arr))
    }


def extract_linguistic_features(texts):
    """Trích xuất các đặc trưng ngôn ngữ học (Linguistic/Handcrafted features) theo docs/feature.md Bước 5."""
    features = []
    punc_regex = re.compile(r'[!?.,;:\'"\(\)\[\]{}]')
    for t in texts:
        if not isinstance(t, str):
            t = str(t)
        char_len = len(t)
        words = t.split()
        word_count = len(words)
        avg_word_len = (sum(len(w) for w in words) / word_count) if word_count > 0 else 0.0
        upper_chars = sum(1 for c in t if c.isupper())
        punc_chars = len(punc_regex.findall(t))
        features.append({
            'char_count': char_len,
            'word_count': word_count,
            'avg_word_length': float(avg_word_len),
            'uppercase_ratio': float(upper_chars / max(1, char_len)),
            'punctuation_ratio': float(punc_chars / max(1, char_len)),
            'digit_fraction': float(bool(re.search(r'\d', t))),
            'negation_cue_count': len(NEGATION_CUES.findall(t)),
            'contrast_cue_count': len(CONTRAST_CUES.findall(t))
        })
    return features


def select_chi2_features(matrix, labels, feature_names, top_k=25):
    """Lựa chọn đặc trưng phân loại bằng kiểm định Chi-Square (chi2) theo docs/feature.md Bước 6."""
    from sklearn.feature_selection import chi2
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or matrix.shape[1] == 0:
        return []
    chi2_scores, p_values = chi2(matrix, labels)
    chi2_scores = np.nan_to_num(chi2_scores, nan=0.0)
    p_values = np.nan_to_num(p_values, nan=1.0)
    k = min(top_k, len(feature_names))
    top_indices = np.argsort(chi2_scores)[-k:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            'term': str(feature_names[idx]),
            'chi2_score': float(chi2_scores[idx]),
            'p_value': float(p_values[idx])
        })
    return results


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
    chars=np.asarray([len(t) for t in text])
    cutoff=int(settings['max_length'])
    quantiles={str(q):float(np.quantile(lengths,q)) for q in (.5,.75,.9,.95,.99,1.)}
    overflow_count=int(np.sum(lengths>cutoff))
    overflow_fraction=overflow_count/len(lengths)
    # Gợi ý chỉ dựa vào tỷ lệ bị cắt trong Train; mặc định không đổi cách tokenize.
    truncation_recommendation=(
        'Review a longer max_length or an explicitly evaluated truncation strategy; '
        'do not change preprocessing based on test results.'
        if overflow_fraction >= .05 else
        'Keep the configured cutoff; the observed train overflow is below 5%. '
        'Do not add a special truncation strategy without evidence.'
    )
    statistics={'train_samples':len(text),'fitted_split':'train',
        'token_quantiles':quantiles,
        'summary_statistics':{
            'words': compute_summary_statistics(words),
            'tokens': compute_summary_statistics(lengths),
            'chars': compute_summary_statistics(chars)
        },
        'truncation_rates':{str(cut):float(np.mean(lengths>cut)) for cut in (64,96,128,256)},
        'configured_max_length':cutoff,'configured_cutoff_overflow_count':overflow_count,
        'configured_cutoff_overflow_fraction':overflow_fraction,
        'configured_cutoff_decision':truncation_recommendation,
        'preserve_negation_punctuation':True,'normalized_train_conflicting_label_groups':0}
    labelsets={}
    for t,y in zip(text,labels):labelsets.setdefault(normalized(t),set()).add(int(y))
    statistics['normalized_train_conflicting_label_groups']=sum(len(v)>1 for v in labelsets.values())
    signals=[]
    cue_rows=[]
    cue_examples=[]
    for label in (0,1):
        chosen=[t for t,y in zip(text,labels) if y==label]
        matches=[m.group(0).casefold() for t in chosen for m in NEGATION_CONTRAST.finditer(t)]
        cue_rows.append({'label':label,'samples':len(chosen),
            'reviews_with_negation_or_contrast':sum(bool(NEGATION_CONTRAST.search(t)) for t in chosen),
            'review_fraction':sum(bool(NEGATION_CONTRAST.search(t)) for t in chosen)/len(chosen),
            'negation_cue_matches':sum(bool(NEGATION_CUES.fullmatch(c)) for c in matches),
            'contrast_cue_matches':sum(bool(CONTRAST_CUES.fullmatch(c)) for c in matches)})
        selected=0
        for i,(t,y) in enumerate(zip(text,labels)):
            if int(y)!=label:continue
            found=[m.group(0) for m in NEGATION_CONTRAST.finditer(t)]
            if found and selected<4:
                cue_examples.append({'index':i,'label':label,'cues':', '.join(found),
                    'text':t[:220]+('...' if len(t)>220 else '')})
                selected+=1
        signals.append({'label':label,'samples':len(chosen),
            'mean_words':float(words[labels==label].mean()),'mean_tokens':float(lengths[labels==label].mean()),
            'negation_or_contrast_fraction':sum(bool(PROTECTED.search(t)) for t in chosen)/len(chosen),
            'digit_fraction':sum(bool(re.search(r'\d',t)) for t in chosen)/len(chosen)})
    save_json(output/'data_quality.json',{'splits':quality,'normalized_cross_split_overlap':overlap,
        'action':'audit only; official splits unchanged; no fitting with heldout text/labels'})
    save_json(output/'preprocessing_statistics.json',statistics)
    save_json(output/'text_signals.json',signals)
    save_json(output/'negation_contrast.json',{'fitted_split':'train','labels':cue_rows,
        'examples':cue_examples,
        'interpretation':'Descriptive association with labels only; counts do not establish causal impact.'})
    _table_image(quality,['split','samples','missing_or_nonstring','empty_or_whitespace','normalized_duplicate_rows','invalid_binary_labels'],
        output/'data_quality.png','Data contract and quality counts; no automatic heldout-driven filtering',widths=[.15,.12,.2,.2,.2,.13])
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for label in (0,1):
        axes[0].hist(words[labels==label],bins=35,alpha=.55,label=f'Label {label}')
        axes[1].hist(lengths[labels==label],bins=35,alpha=.55,label=f'Label {label}')
    axes[0].set(xlabel='Whitespace words',ylabel='Train examples')
    axes[1].axvline(settings['max_length'],ls='--',color='black',label='Configured cutoff')
    axes[1].set(xlabel='WordPiece tokens incl. special tokens',ylabel='Train examples')
    axes[1].axvspan(settings['max_length'],max(settings['max_length']+1,int(lengths.max())),
        color='#d9534f',alpha=.09,label='Region truncated by tokenizer')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    finish(fig,output/'lengths_by_class.png')
    # Bivariate analysis: Boxplot so sánh phân bố độ dài từ & tokens theo nhãn
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    axes[0].boxplot([words[labels==0],words[labels==1]],tick_labels=['Negative (0)','Positive (1)'],patch_artist=True,
                    boxprops=dict(facecolor='#d9edf7',color='#31708f'),
                    medianprops=dict(color='#a94442',linewidth=1.5))
    axes[0].set(title='Whitespace Words Distribution by Label',ylabel='Word count')
    axes[0].grid(axis='y',alpha=0.2)
    axes[1].boxplot([lengths[labels==0],lengths[labels==1]],tick_labels=['Negative (0)','Positive (1)'],patch_artist=True,
                    boxprops=dict(facecolor='#dff0d8',color='#3c763d'),
                    medianprops=dict(color='#a94442',linewidth=1.5))
    axes[1].axhline(cutoff,color='red',linestyle='--',label=f'Cutoff ({cutoff})')
    axes[1].set(title='WordPiece Tokens Distribution by Label',ylabel='Token count')
    axes[1].legend()
    axes[1].grid(axis='y',alpha=0.2)
    finish(fig,output/'lengths_boxplot.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].bar(list(statistics['truncation_rates']),[100*v for v in statistics['truncation_rates'].values()])
    axes[0].set(xlabel='Token cutoff',ylabel='Train truncated (%)',title='Cutoff sensitivity, train only')
    axes[1].bar(['Negative','Positive'],[100*r['negation_or_contrast_fraction'] for r in signals])
    axes[1].set(ylabel='Train reviews (%)',title='Negation/contrast: preserve linguistic cues')
    finish(fig,output/'preprocessing_diagnostics.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    x=np.arange(2)
    axes[0].bar(x,[r['review_fraction']*100 for r in cue_rows],color=['#b44','#387'])
    axes[0].set(xticks=x,xticklabels=['Negative (0)','Positive (1)'],ylabel='Reviews (%)',
        title='Reviews containing ≥1 cue')
    axes[1].bar(x-.18,[r['negation_cue_matches'] for r in cue_rows],.36,label='Negation')
    axes[1].bar(x+.18,[r['contrast_cue_matches'] for r in cue_rows],.36,label='Contrast')
    axes[1].set(xticks=x,xticklabels=['Negative (0)','Positive (1)'],ylabel='Cue matches',
        title='Total cue occurrences')
    axes[1].legend()
    for ax in axes:ax.grid(axis='y',alpha=.2)
    finish(fig,output/'negation_contrast_by_label.png')
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
    means_by_label=[]
    for label in (0,1):
        means=np.asarray(matrix[labels==label].mean(axis=0)).ravel()
        means_by_label.append(means)
        for index in np.argsort(means)[-25:][::-1]:
            terms.append({'label':label,'term':str(names[index]),'mean_tfidf':float(means[index])})
    delta=means_by_label[1]-means_by_label[0]
    ordered_delta=np.argsort(delta)
    delta_indices=(list(ordered_delta) if len(ordered_delta)<=30 else
        list(ordered_delta[:15])+list(ordered_delta[-15:][::-1]))
    delta_terms=[{'term':str(names[i]),'mean_tfidf_negative':float(means_by_label[0][i]),
        'mean_tfidf_positive':float(means_by_label[1][i]),
        'delta_positive_minus_negative':float(delta[i])} for i in delta_indices]
    features={'fitted_split':'train','settings':feature_settings,'train_shape':list(matrix.shape),'nnz':int(matrix.nnz),
        'density':float(matrix.nnz/(matrix.shape[0]*matrix.shape[1])),
        'sparse_array_storage_bytes':int(matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes),
        'storage_note':'array footprint only, not process RAM peak','terms':terms,
        'class_delta_terms':delta_terms,
        'class_delta_interpretation':'Descriptive mean TF-IDF difference; not causal and not classifier coefficients.'}
    save_json(output/'feature_summary.json',features)
    fig,axes=plt.subplots(1,2,figsize=(15,9))
    for label,ax in enumerate(axes):
        group=[r for r in terms if r['label']==label][::-1]
        ax.barh([r['term'] for r in group],[r['mean_tfidf'] for r in group])
        ax.set(title=f'Train mean TF-IDF, label {label} (top 25)',xlabel='Mean feature value')
        ax.tick_params(axis='y',labelsize=8)
    finish(fig,output/'tfidf_features.png')
    selected=sorted(delta_terms,key=lambda row:row['delta_positive_minus_negative'])
    fig,ax=plt.subplots(figsize=(10,7))
    ax.barh([r['term'] for r in selected],[r['delta_positive_minus_negative'] for r in selected],
        color=['#b44' if r['delta_positive_minus_negative']<0 else '#387' for r in selected])
    ax.axvline(0,color='black',linewidth=.8)
    ax.set(xlabel='Mean TF-IDF: Positive − Negative',title='Train-only features with largest class differences')
    ax.grid(axis='x',alpha=.2)
    finish(fig,output/'tfidf_class_delta.png')
    # Trích xuất và trực quan hóa đặc trưng ngôn ngữ học (Linguistic Features)
    ling_feats = extract_linguistic_features(text)
    ling_summary = {}
    for key in ['char_count', 'word_count', 'avg_word_length', 'uppercase_ratio', 'punctuation_ratio', 'digit_fraction', 'negation_cue_count', 'contrast_cue_count']:
        vals_0 = [f[key] for f, y in zip(ling_feats, labels) if y == 0]
        vals_1 = [f[key] for f, y in zip(ling_feats, labels) if y == 1]
        m0 = float(np.mean(vals_0)) if vals_0 else 0.0
        m1 = float(np.mean(vals_1)) if vals_1 else 0.0
        all_vals = vals_0 + vals_1
        ling_summary[key] = {
            'label_0_mean': m0,
            'label_1_mean': m1,
            'overall_mean': float(np.mean(all_vals)) if all_vals else 0.0
        }
    save_json(output/'linguistic_features.json', {
        'fitted_split': 'train',
        'metrics': ling_summary,
        'description': 'Linguistic and handcrafted features extracted according to docs/feature.md Step 5'
    })
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ratios = ['uppercase_ratio', 'punctuation_ratio', 'avg_word_length']
    titles = ['Uppercase Ratio', 'Punctuation Ratio', 'Avg Word Length (chars)']
    for ax, r_key, tit in zip(axes, ratios, titles):
        means = [ling_summary[r_key]['label_0_mean'], ling_summary[r_key]['label_1_mean']]
        bars = ax.bar(['Negative (0)', 'Positive (1)'], means, color=['#b44', '#387'])
        ax.bar_label(bars, fmt='%.3f' if 'ratio' in r_key else '%.2f')
        ax.set(title=tit, ylabel='Mean value')
        ax.grid(axis='y', alpha=0.2)
    finish(fig, output/'linguistic_features.png')

    # Lựa chọn đặc trưng phân loại bằng kiểm định Chi-Square
    chi2_terms = select_chi2_features(matrix, labels, names, top_k=25)
    features['chi2_selected_terms'] = chi2_terms
    save_json(output/'chi2_features.json', {
        'fitted_split': 'train',
        'top_k': len(chi2_terms),
        'terms': chi2_terms,
        'method': 'Chi-Square (chi2) test on train TF-IDF matrix vs sentiment labels'
    })
    save_json(output/'feature_summary.json', features)
    fig, ax = plt.subplots(figsize=(10, max(4, int(0.28 * len(chi2_terms) + 2))))
    if chi2_terms:
        rev_chi2 = chi2_terms[::-1]
        ax.barh([r['term'] for r in rev_chi2], [r['chi2_score'] for r in rev_chi2], color='#2b5c8f')
        ax.set(xlabel='Chi-Square Score (Higher = Stronger Association with Sentiment)',
               title='Top TF-IDF Features Selected by Chi-Square Test (Train only)')
    else:
        ax.text(0.5, 0.5, 'Insufficient distinct features/classes for Chi-Square test', ha='center', va='center')
    ax.grid(axis='x', alpha=0.2)
    finish(fig, output/'chi2_features.png')
    return statistics,features
