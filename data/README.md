# data/

Git ignores every file in this folder except this README. Do not commit comment data or model output:
the comments contain offensive text.

## Real data: Jigsaw Toxic Comment Classification Challenge

| Item | Value |
|---|---|
| Name | Jigsaw Toxic Comment Classification Challenge (Wikipedia talk-page comments) |
| Source | <https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data> |
| Terms | Accept the competition rules on Kaggle. The comment text is under CC0, the labels follow the competition rules |
| Files | `train.csv`, `test.csv`, `test_labels.csv` |

| File | Columns |
|---|---|
| `train.csv` | `id`, `comment_text`, `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate` (0/1) |
| `test.csv` | `id`, `comment_text` |
| `test_labels.csv` | `id` and the 6 labels. The value `-1` means "not scored": the loader drops these rows |

Set `KINDIFY_DATA=data/train.csv`, `KINDIFY_TEST_DATA=data/test.csv` and `KINDIFY_TEST_LABELS=data/test_labels.csv`.
With the official test files, the test part is the scored official test set at its natural toxic share.
An optional `lang` column gives a per-language table in the evaluation.

## Other useful sets (not used by the code by default)

| Set | Use |
|---|---|
| Jigsaw Multilingual Toxic Comment Classification (2020), Kaggle | Multilingual validation data (`lang` column) |
| Jigsaw Unintended Bias in Toxicity Classification, Kaggle | Identity columns for the bias metrics |
| ParaDetox (`s-nlp/paradetox`) | Toxic and polite sentence pairs to evaluate rewrites |

## Synthetic data (no download)

`kindify synth --rows 4000 --out data/synthetic_comments.csv` writes fake talk-page comments with the same
columns. The toxic comments contain mild insults only, and the generator writes no hate speech.
With no `--data` and no `KINDIFY_DATA`, `train` generates them in memory.
