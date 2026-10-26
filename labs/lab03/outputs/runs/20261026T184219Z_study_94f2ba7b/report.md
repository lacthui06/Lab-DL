# Fine-tuning run report

Run: `20261026T184219Z_study_94f2ba7b`

Official splits: 8,530 train / 1,066 validation / 1,066 test.

## Test metrics

```json
{
  "accuracy": 0.8339587242026266,
  "binary_precision": 0.8384030418250951,
  "binary_recall": 0.8273921200750469,
  "binary_f1": 0.8328611898016998,
  "macro_precision": 0.8340163357273623,
  "macro_recall": 0.8339587242026267,
  "macro_f1": 0.8339515641459571
}
```

## Qualitative error review

The examples below come from this run. Linguistic explanations are hypotheses, not measured model causes.

- Review: a very familiar tale , one that's been told by countless filmmakers about italian- , chinese- , irish- , latin- , indian- , russian- and other hyphenate american young men struggling to balance conflicting cultural messages .
  True=0, predicted=1, confidence=0.999.
  Hypothesis: implicit sentiment, irony or contextual references need a manual reading of this review. This is a linguistic hypothesis, not a causal attribution.
- Review: entertains not so much because of its music or comic antics , but through the perverse pleasure of watching disney scrape the bottom of its own cracker barrel .
  True=0, predicted=1, confidence=0.999.
  Hypothesis: contrast or concession may mix positive and negative cues; negation scope may reverse the sentiment of nearby words. This is a linguistic hypothesis, not a causal attribution.
- Review: its spirit of iconoclastic abandon -- however canned -- makes for unexpectedly giddy viewing .
  True=1, predicted=0, confidence=0.999.
  Hypothesis: contrast or concession may mix positive and negative cues. This is a linguistic hypothesis, not a causal attribution.
- Review: a soap-opera quality twist in the last 20 minutes . . . almost puts the kibosh on what is otherwise a sumptuous work of b-movie imagination .
  True=1, predicted=0, confidence=0.999.
  Hypothesis: implicit sentiment, irony or contextual references need a manual reading of this review. This is a linguistic hypothesis, not a causal attribution.
