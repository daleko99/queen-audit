# findings on Queen

these are measurements of the released model as well as some ways to make it more accurate.

all numbers come from paired comparisons on 600 held-out middlegame or endgame positions unless stated otherwise. brackets are 95% bootstrap confidence intervals, grouped by position. "fact errors" are counted by the deterministic checker (`datagen/tree/facts.py`), it checks for illegal moves, wrong material counts, evaluations that stockfish disagrees with and best or critical lines that stockfish disagrees with.

## 1) how often the released model is wrong

2.72 fact errors per analysis on average. in over 8000 samples:

- 5.0% of the moves are illegal
- 7.8% of critical-line moves are mistakes
- 22.5% of evaluations are off by 10 or more win rate points
- 67.4% of "extra piece" claims were false
- 38.8% of "who is ahead in material" claims were false

## 2) a sign bug in the self-distillation parser

the papers self distillation code reads the models evaluation sentence to decide which move to consolidate. the parser had no pattern for " is up / down N pawns " so "Black is up 2 pawns" became a white POV by default, and "the player is up N" could be flipped later

- 309 of 1077 evaluations were read with the wrong sign. once a verb aware pattern was put in place, only 1 remained
- reparsing the self distillation data with the corrected parser flips 23/110 root and 34/330 child signs: 11/110 steps select a different child or recursion verdict and changes the move in 3/72 accepted steps

## 3) the model can not see the move number

the board encoders input planes never carry a move number so the model guesses move numbers and is often wrong. by mapping each move number onto the true move number via ply (`datagen/tree/move_numbers.py`), judge complaints about move numbers reduce from 299/600 analyses to 8, and raise its structure score by +0.11 [+0.08, +0.16]

## 4) deterministic correction beats more sampling

`datagen/tree/fact_correct.py` corrects wrong material claims, removes illegal lines at the last legal move and replaces the evaluation with a stockfish one.

`datagen/tree/best_of_k.py` picks the sample with the fewest fact errors out of K, then corrects and renumbers it

- k=8 picks the cleanest output, corrects the factual errors and renumbers moves. this cut the fact errors from 2.72 to 0.30 per analysis. an opus 5.5 judge rates it +0.21 on structure and +0.38 on concepts above released
- k=1 with correction matches every K on the judge (2400 judgements). more samples only lower residual fact errors (0.61 at K=1, 0.30 at K=8). deterministic correction is the primary driver of the gain, not selection.

## 5) training on the models own corrected output

self distillation rounds using best of 8 selection and correction, then supervised fine tuning of the bridge:

- round 1: -0.16 [-0.32, -0.01] fact errors vs released. selection alone gives 65% cleaner training data but only ~4% cleaner outputs. selection with correction doubles this effect (-0.25 [-0.40, -0.10] against randomly chosen control, +5.3% fully clean analyses) without dropping content
- round 3 was trained on 2000 fresh positions and led to -26% in fact errors vs released (-0.70). scaling on just 400 positions only led to -0.14, so data per round drives it
- round 4 adds nothing (+0.00 fact errors, judge within ±0.02). material claims still improve, but illegal line moves and critical line mistakes never do.
- two epochs led to no gain.
- tripling the learning rate led to worse performance.
- adding decoder LoRA gained -0.12 fact errors against bridge-only after correction. no changes on the opus judge.

## 6) DPO with checker labels

DPO on pairs labelled by the motif claim checker (fewer vs more refuted claims), four seeds, one epoch (336 steps):

- refuted motif claims went from 54.7% to 33.0-40.8% (mean -17.5 points). every seeds interval was below zero
- the effect grows with steps (56 steps: -2.2, 112: -5.7, 224: -12.7 points), but 1.5 epochs is already half broken and 2 epochs fully collapses into repeated tokens
- the explanations dont get better though. the opus judge scores all four seeds slightly below released on concepts (-0.07 to -0.15) and structure (-0.08 to -0.15). the model drops claims instead of fixing them
- DPO on fact error pairs instead cuts checkable content by 24% with no change in errors
