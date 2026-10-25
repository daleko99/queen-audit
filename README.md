# queen-audit

this repo measures the model "Queen" released with "Language Models that Play Chess and Explain Their Moves" (Bhaskar, Cheng, Chen. [arXiv:2610.03695](https://arxiv.org/abs/2610.03695), [code](https://github.com/queen-project/queen)) and has the tools to measure and correct it

| headline results (details in [FINDINGS.md](FINDINGS.md)): |
|---|
| Queen makes an average of 2.72 checkable fact errors per analysis. things such as illegal moves in its lines, wrong material counts or evaluations that disagree with stockfish |
| a deterministic corrector brings that to 0.61 with one sample (0.30 if u pick the best of 8), without retraining |
| the model cannot see the move number. a ply-based fix helps this in 291 of 299 affected cases |
| a sign bug in the original self distillation parser misread 309 of 1077 evaluations. for example: "black is up 2 pawns" read as white being up 2 pawns instead. |
| self distillation on the corrected output reduced fact errors by 26% by round 3 before plateauing. DPO reduced refuted claims by 17 points but by saying less, not by being right more often |

> this project is independent. i am not affiliated or endorsed by the Queen authors or by princeton university.

## tools

| file | |
|---|---|
| [`datagen/tree/facts.py`](datagen/tree/facts.py) | fact checker - checks move legality across text and data fields, material claims, evaluation against stockfish, and cross references best move and critical line vs stockfish |
| [`datagen/tree/fact_correct.py`](datagen/tree/fact_correct.py) | deterministic corrector - rewrites wrong material claims, removes illegal move branches and replaces the model evaluation errors with stockfish |
| [`datagen/tree/move_numbers.py`](datagen/tree/move_numbers.py) | maps guessed move count onto the real one via ply count |
| [`datagen/tree/best_of_k.py`](datagen/tree/best_of_k.py) | chooses fewest error sample of K, then corrects and renumbers it |
| [`datagen/tree/fact_sft.py`](datagen/tree/fact_sft.py) | builds best-of-k, control, and fact error sets from k-sample outputs |
| [`eval/facts_eval.py`](eval/facts_eval.py) | runs stockfish and the checker over analysis |

the tools use Queen's analysis format (`<PIECE_MN> <SQUARE_7> <SQUARE_22>`)

## Setup

```bash
pip install -r requirements.txt
python -m pytest
```

`eval/facts_eval.py audit` needs a stockfish binary on the PATH

## reproduction

for reproduction, the following is included:

* `data/test_positions.jsonl` includes the 600 test positions
* the fact checker, corrector, move number fix, and best of k tools.
* `eval/facts_eval.py` for running the checks with stockfish

on the contrast, some things are not included:

* Queens raw outputs because the weights have no license
* the training code for sections 5 and 6
* the AI judge used for structure and concept scores

this means the training results and judge scores cant be rerun from this repo alone, however their setups are described in [FINDINGS.md](FINDINGS.md)

to redo the error numbers:

* run Queen on the 600 positions
* run `facts_eval audit` and then `check` on its outputs

this will give u the fact error numbers in sections 1 and 4. sampling is random so the numbers wont be exact but should be similar

## license

apache-2.0

see [LICENSE](LICENSE) and [NOTICE](NOTICE)

parts of `datagen/self_distill/` and `eval/` come from the Queen project. NOTICE lists which files.
