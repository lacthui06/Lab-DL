# Fine-tuning run report

Run: `20261027T083010Z_study_99bce559`

Official splits: 8,530 train / 1,066 validation / 1,066 test.

## Test metrics

```json
{
  "accuracy": 0.8367729831144465,
  "binary_precision": 0.8269581056466302,
  "binary_recall": 0.851782363977486,
  "binary_f1": 0.8391866913123844,
  "macro_precision": 0.837076731740143,
  "macro_recall": 0.8367729831144466,
  "macro_f1": 0.8367362027990494
}
```

## Qualitative error review

The examples below come from this run. Linguistic explanations are hypotheses, not measured model causes.

- Review: hey arnold ! is now stretched to barely feature length , with a little more attention paid to the animation . still , the updated dickensian sensibility of writer craig bartlett's story is appealing .
  True=0, predicted=1, confidence=0.998.
  Hypothesis: negation scope may reverse the sentiment of nearby words. This is a linguistic hypothesis, not a causal attribution.
- Review: though its rather routine script is loaded with familiar situations , the movie has a cinematic fluidity and sense of intelligence that makes it work more than it probably should .
  True=0, predicted=1, confidence=0.998.
  Hypothesis: implicit sentiment, irony or contextual references need a manual reading of this review. This is a linguistic hypothesis, not a causal attribution.
- Review: the wild thornberrys movie has all the sibling rivalry and general family chaos to which anyone can relate .
  True=1, predicted=0, confidence=0.997.
  Hypothesis: implicit sentiment, irony or contextual references need a manual reading of this review. This is a linguistic hypothesis, not a causal attribution.
- Review: a movie where story is almost an afterthought amidst a swirl of colors and inexplicable events .
  True=1, predicted=0, confidence=0.997.
  Hypothesis: implicit sentiment, irony or contextual references need a manual reading of this review. This is a linguistic hypothesis, not a causal attribution.
