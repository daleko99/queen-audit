# findings on Queen

these are measurements of the released model as well as some ways to make it more accurate.

all numbers come from paired comparisons on 600 held-out middlegame or endgame positions unless stated otherwise. brackets are 95% bootstrap confidence intervals, grouped by position. "fact errors" are counted by the deterministic checker (`datagen/tree/facts.py`), it checks for illegal moves, wrong material counts, evaluations that stockfish disagrees with and best or critical lines that stockfish disagrees with.

## how to read judge scores

the original paper used an LLM judge to score structure, concepts and fluency from 1 to 5 in each analysis produced by queen. the judge used here is opus 5.5 whereas the paper used gpt 5.6 sol (high), and our judge was harsher so scores cant be compared with the paper (released model got: 3.22 structure, 1.69 concepts, 4.00 fluency from opus 5.5, whereas it got: 3.51 structure, 2.76 concepts and 4.45 from gpt 5.6 sol (high) in the paper). 

the meanings of the categories are:
- structure refers to whether it reads as a coherent analysis with key features, candidate moves and a final verdict, without contradicting itself.
- concepts refer to chess claims and whether they are correct, such as motifs, piece/square references, move legality and evaluations. 
- fluency is just the writing itself. 

the judge is also given stockfish to cross reference against. for scale, concept scores are near the floor (95% of released analyses get a 1 or 2) so a +0.38 is a big jump whereas structure scores start much higher (3.22) so a +0.11 is small. anything within about ±0.02 is pretty much no change.


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

## 7) what a judge sees
- an opus 5.5 judge's concept scores for the released model sit at the floor, at 95% of analyses getting 1 or 2 out of 5, so the judge cant tell analyses with motif claim errors from clean ones (1.68 vs 1.71).
- when the judge explains its score, it mentions incorrect evaluations (88%), wrong material counts (72%), move numbering (60%), wrong best moves (56%), illegal moves (44%), and motif claim errors (18%). this shows us that motif claim errors are only a small portion of what the judge dislikes. note that these are counted by a keyword search, so its rough and a mention isnt always a real fault. material, move legality and evaluation are cheap to verify against the board and stockfish (which is what the fact checker does)
- part of the judge's penalty is for move numbers, which the model cannot observe (section 3)

## 8) reproduction cost
- the paper's stage 1-4 training schedule measured on an A100 80GB is 26.8 sequences/s compiled with 48% being padding, so about 640~ A100 hours before self distillation is needed. this was estimated by extrapolating 10 steps of stage 1 QA data to stages 1-4
