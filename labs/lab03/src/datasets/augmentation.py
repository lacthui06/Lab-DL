"""Tạo biến thể train có kiểm soát; giữ nguyên validation/test và văn bản gốc."""
import hashlib
import json
import math
import random
import re
from datasets import DatasetDict, Dataset, concatenate_datasets

# Từ thay thế cùng cực tính, một từ mỗi mẫu; không thay tên riêng/stopword ngẫu nhiên.
LEXICON = {
    1: {'wonderful':'excellent', 'excellent':'wonderful', 'enjoyable':'entertaining',
        'entertaining':'enjoyable', 'charming':'delightful', 'delightful':'charming',
        'funny':'amusing', 'amusing':'funny', 'superb':'excellent', 'hilarious':'very funny'},
    0: {'terrible':'awful', 'awful':'terrible', 'boring':'tedious', 'tedious':'boring',
        'dull':'boring', 'unfunny':'humorless', 'humorless':'unfunny',
        'disappointing':'unsatisfying', 'pointless':'purposeless'},
}
PROTECTED = re.compile(r"\b(?:not|no|never|neither|nor|hardly|barely|without|but|however|although|despite|yet)\b|n['’]t\b",re.I)


def normalized(text):
    return ' '.join(text.casefold().split())


def augment_train(raw, settings=None, seed=42):
    settings = settings or {}
    if not settings.get('enabled',False):
        return raw, {'enabled':False,'original_train_size':len(raw['train']),'effective_train_size':len(raw['train']),'added_count':0,'records':[]}
    ratio=settings.get('max_ratio',.25)
    if isinstance(ratio,bool) or not isinstance(ratio,(int,float)) or not math.isfinite(ratio) or not 0<ratio<=.5:
        raise ValueError('augmentation.max_ratio must be finite and in (0, 0.5]')
    train=raw['train']
    originals=list(train['text']);labels=list(map(int,train['label']))
    if set(labels)!={0,1}:raise ValueError('Augmentation requires both binary classes')
    occupied=set(map(normalized,originals))
    # Chỉ đọc heldout text để chặn trùng; không dùng nhãn hay metric heldout.
    heldout={normalized(text) for split in ('validation','test') for text in raw[split]['text']}
    candidates={0:[],1:[]};seen=set(occupied);blocked=0;protected=0
    rng=random.Random(seed)
    order=list(range(len(train)));rng.shuffle(order)
    for index in order:
        text=originals[index];label=labels[index]
        if PROTECTED.search(text):protected+=1;continue
        options=[m for m in re.finditer(r'\b[a-z]+\b',text) if m.group() in LEXICON[label]]
        if not options:continue
        rng.shuffle(options)
        for match in options:
            replacement=LEXICON[label][match.group()]
            generated=text[:match.start()]+replacement+text[match.end():]
            key=normalized(generated)
            if key in seen:continue
            if key in heldout:blocked+=1;continue
            seen.add(key)
            candidates[label].append({'parent_index':index,'original_text':text,'text':generated,'label':label,
                'source_word':match.group(),'replacement':replacement,'word_start':match.start(),'word_end':match.end(),'method':'single_same_polarity_lexical_substitution_v1'})
            break
    cap=int(min(labels.count(0),labels.count(1))*ratio)
    per_class=min(cap,len(candidates[0]),len(candidates[1]))
    records=candidates[0][:per_class]+candidates[1][:per_class];rng.shuffle(records)
    if not records and settings.get('require_added',True):
        raise ValueError('No balanced augmentation candidates; inspect dataset/lexicon before GPU training')
    data=DatasetDict(raw)
    if records:
        added=Dataset.from_dict({'text':[r['text'] for r in records],'label':[r['label'] for r in records]},features=train.features)
        data['train']=concatenate_datasets([train,added])
    digest=hashlib.sha256(json.dumps(records,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    summary={'enabled':True,'method':'single_same_polarity_lexical_substitution_v1','seed':seed,
        'original_train_size':len(train),'effective_train_size':len(data['train']),'added_count':len(records),
        'added_per_class':{'0':per_class,'1':per_class},'maximum_requested_ratio':ratio,
        'eligible_per_class':{str(c):len(candidates[c]) for c in (0,1)},'protected_reviews_skipped':protected,
        'heldout_text_collisions_rejected':blocked,'validation_and_test_modified':False,
        'label_policy':'inherit parent label; approximate semantic preservation, not human-verified new ground truth',
        'records_sha256':digest,'records':records}
    return data,summary


def export_augmentation(summary, output):
    """Lưu cả records để review; ảnh hiển thị từ cùng files này trong notebook."""
    from ..utils.runs import save_json
    from ..utils.plots import finish
    from ..analysis import _table_image
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/'augmentation.json',summary)
    records=summary['records']
    if not summary['enabled']:return
    fig,ax=plt.subplots(figsize=(7,4))
    bars=ax.bar(['Official train','Synthetic added','Effective train'],[summary['original_train_size'],len(records),summary['effective_train_size']])
    ax.bar_label(bars);ax.set(ylabel='Examples',title='Training only: original and synthetic counts')
    finish(fig,output/'augmentation_counts.png')
    _table_image(records[:8],['original_text','text','label','source_word','replacement'],output/'augmentation_examples.png',
        'Synthetic train variants: inherited labels need semantic review',widths=[.35,.35,.06,.12,.12])
