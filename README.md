# CSE 256 PA2 — Transformer Blocks and Pythia Analysis

Mansi Dhamne

## Setup

Use Python 3.9 or later. Run these commands from this directory to create an
isolated environment and install dependencies using the same interpreter that
will run the code.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab', download_dir='nltk_data', raise_on_error=True)"
```

The tokenizer searches this directory's `nltk_data` folder automatically.

## Run Parts 1 and 2

```bash
python main.py                 # Part 1 with default hyperparameters
python main.py --part part1    # Encoder and speaker classifier
python main.py --part part2    # Causal decoder language model
```

Dataset paths resolve relative to `main.py`. By default, results are written
beside the code in `outputs/part1/` or `outputs/part2/`.

```bash
python main.py --part part2 --device cpu --threads 2 --output-dir outputs/my_part2_run
python main.py --help
```

`--device auto` selects CUDA when available and CPU otherwise. Explicit options
are `cpu`, `cuda`, and `mps`, with the latter two requiring compatible hardware
and PyTorch builds. The default CPU thread count is 2.

## Files

| File | Purpose |
| --- | --- |
| `main.py` | Data loading, training, evaluation, command-line options |
| `transformer.py` | Attention, encoder, classifier, causal decoder |
| `dataset.py` | Supplied classification and language-model datasets |
| `tokenizer.py` | Supplied NLTK word tokenizer with stable vocabulary IDs |
| `utilities.py` | Attention checks, annotated heatmaps, numeric exports |
| `Part_3.ipynb` | Executed Part 3 notebook with answers, figures, and outputs |
| `speechesdataset/` | Supplied training and test data |
| `outputs/` | Recorded metrics, attention figures, and numeric attention matrices |

## Default architecture and training settings

Both models use learned token and absolute position embeddings of width 64,
four Transformer blocks, two heads per block, residual connections,
pre-layer normalization, and a final layer norm. Attention is implemented
using projections, scaled dot products, masking, and softmax. All vocabulary
IDs are assigned deterministically from the training files only. The vocabulary
contains 5,755 entries, including padding and unknown tokens.

| Setting | Part 1 | Part 2 |
| --- | --- | --- |
| Seed | 42 | 42 |
| Batch size | 16 | 16 |
| Block size | 32 | 32 |
| Optimizer | Adam, default betas | Adam, default betas |
| Learning rate | 0.001 | 0.001 |
| Training duration | 15 epochs | 500 updates |
| Block FFN | 64 → 256 → 64, ReLU | 64 → 100 → 64, ReLU |
| Dropout / weight decay | 0 / 0 | 0 / 0 |

Part 1 averages the encoder outputs over non-padding words, then applies a
64 → 100 → 3 ReLU classifier. Cross-entropy trains the encoder and classifier
jointly. The encoder FFN width of 256 is an implementation choice because the
handout leaves this width unspecified. The supplied classification test file
is `test_CLS.tsv`. Test accuracy is measured after every epoch, and epoch 15
is the final model. Encoder parameters total 570,432 and classifier parameters
6,803. The recorded final accuracy is 84.53%.

Part 2 masks future positions before softmax and uses an untied output
projection. Training uses shuffled stride-one next-token windows from
`train_LM.txt`, for 256,000 target presentations. Every 100 updates, evaluation
scores every next-token target once in consecutive blocks of up to 32 targets.
Context and positions reset at each block boundary. The final partial block
is padded, and padded targets are excluded. Perplexity is
`exp(total negative log likelihood / valid target count)`. The decoder contains
864,011 parameters. Recorded final perplexities are 170.46 on training,
370.49 on Obama, 486.06 on George W. Bush, and 409.90 on George H. W. Bush.

## Part 1 results

The final test accuracy after 15 epochs is **84.53%**. The encoder contains
**570,432 parameters**, and the classifier adds **6,803 parameters**. Test
accuracy peaks at 85.07% at epoch 13 and finishes at 84.53% at epoch 15.

| Epoch | Test accuracy (%) |
| --- | ---: |
| 1 | 40.67 |
| 2 | 48.67 |
| 3 | 57.07 |
| 4 | 63.73 |
| 5 | 68.80 |
| 6 | 76.80 |
| 7 | 81.47 |
| 8 | 82.00 |
| 9 | 81.87 |
| 10 | 84.53 |
| 11 | 84.27 |
| 12 | 84.27 |
| 13 | 85.07 |
| 14 | 84.53 |
| 15 | 84.53 |

![Part 1 test accuracy by epoch](outputs/part1/accuracy.png)

The following attention maps show encoder layer 1, head 1 for two sentences.
Query words appear along rows and key words along columns.

![Encoder attention for I will invest in early childhood education](outputs/part1/sentence_1_layer1_head1.png)

![Encoder attention for We have the power to make the world we seek](outputs/part1/sentence_2_layer1_head1.png)

In the first example, “will” attends strongly to “invest” (0.829), while
“childhood” attends mainly to itself (0.646). In the second example, the first
“the” attends to “make” (0.787), and “world” attends to the period (0.945).
Attention occurs on both sides of the diagonal, consistent with bidirectional
encoder attention. The displayed row sums are approximately 1. Cell values
are rounded to three decimals, which can slightly change their displayed sum.

Full-precision attention values and row sums are available in the
[first attention CSV](outputs/part1/sentence_1_layer1_head1.csv) and
[second attention CSV](outputs/part1/sentence_2_layer1_head1.csv).
The complete epoch measurements are in [metrics.csv](outputs/part1/metrics.csv).

## Part 2 results

The decoder contains **864,011 parameters**. After 500 updates, training
perplexity is **170.46**. Final test perplexities are **370.49** for Obama,
**486.06** for George W. Bush, and **409.90** for George H. W. Bush.
Iteration 500 also serves as the final evaluation.

| Iteration | Training | Obama | George W. Bush | George H. W. Bush |
| --- | ---: | ---: | ---: | ---: |
| 100 | 583.78 | 694.65 | 815.28 | 735.15 |
| 200 | 448.01 | 580.87 | 688.90 | 611.44 |
| 300 | 307.78 | 465.13 | 571.34 | 493.65 |
| 400 | 222.49 | 396.83 | 516.62 | 433.98 |
| 500 | 170.46 | 370.49 | 486.06 | 409.90 |

![Training and test perplexities every 100 iterations](outputs/part2/perplexity.png)

Perplexity decreases throughout training for all four datasets. Differences
in vocabulary, topics, recurring phrases, and sentence structure can make
the politicians' test corpora differently predictable under the shared model.
Obama has the lowest final test perplexity and George W. Bush the highest.
These measurements establish the ranking for the supplied test files.

![Decoder attention for I will invest in early childhood education](outputs/part2/sentence_1_layer1_head1.png)

![Decoder attention for We have the power to make the world we seek](outputs/part2/sentence_2_layer1_head1.png)

Both decoder plots show layer 1, head 1. The upper triangle is zero because
the causal mask prevents each query from attending to future tokens. The first
token attends entirely to itself with probability 1. Later positions distribute
attention over the available prefix. For example, “invest” attends to “will”
with probability 0.532 in the first sentence. In the second sentence, the first
“the” attends to “have” with probability 0.815. Every displayed row sums to
approximately 1, as required for normalized attention probabilities.

Full-precision attention values and row sums are available in the
[first attention CSV](outputs/part2/sentence_1_layer1_head1.csv) and
[second attention CSV](outputs/part2/sentence_2_layer1_head1.csv).
The complete perplexity measurements are in [metrics.csv](outputs/part2/metrics.csv).

## Generated outputs and attention checks

Each training command creates:

- `metrics.csv` with epoch accuracies or perplexities every 100 updates.
- `results.json` with settings, parameter counts, measurements, and attention checks.
- `part1_model.pt` or `part2_model.pt` with weights, vocabulary IDs, and configuration.
- Two attention PNGs, matching `.pt` tensors, and CSVs with probabilities and row sums.

For each example sentence, `utilities.py` checks all eight layer/head maps for
row sums of 1 within tolerance `1e-5` and zero padding-key attention. For the
decoder it also checks zero attention above the diagonal. The plotted example
is layer 1, head 1. Each cell displays its probability to three decimals, and
row sums use the unrounded values. The examples are “I will invest in early
childhood education.” and “We have the power to make the world we seek.”

Checkpoints and generated JSON summaries are excluded from Git by `.gitignore`.
They are regenerated by the training commands. Recorded CSVs and figures remain
available in the repository.

## Part 3

Open `Part_3.ipynb` in Jupyter or Google Colab. It includes the executed outputs.
For a fresh run, install the additional notebook dependencies:

```bash
python -m pip install transformers safetensors jupyter nbconvert
```

Run the cells in order. In Colab, select a GPU runtime and upload
`speechesdataset/test_LM_obama.txt` to the working directory. Locally, open the
notebook from this directory so its relative dataset path resolves.

The notebook downloads six checkpoints of `EleutherAI/pythia-160m-deduped`
(about 4 GB). It uses float32 and selects CUDA when available. Its logit-lens
accuracy and transplant loss evaluations cover every next-token target in
consecutive 256-token blocks, resetting context at each block boundary. The
analysis code checks final-layer normalization and restores original component
weights after each transplant.
