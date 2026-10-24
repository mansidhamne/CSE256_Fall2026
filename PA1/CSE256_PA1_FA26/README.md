# CSE 256 Fall 2026 — PA1

Binary sentiment classification on the Rotten Tomatoes sentence data in `data/`. Run every command from the home directory.

Requirements: Python 3.6 or later, PyTorch, NumPy, scikit-learn, and matplotlib.

## Provided baseline

```bash
python main.py --model BOW
```

Trains the supplied bag-of-words networks for 100 epochs and saves `train_accuracy.png` and `dev_accuracy.png`.


## Part 1: Deep Averaging Network

```bash
python main.py --model DAN
```

Trains the final word-level DAN (300-d GloVe, one hidden layer of size 300, ReLU, no dropout, Adam at 0.001, 8 epochs) and prints train and dev accuracy and loss after every epoch. 


```bash
python main.py --model DAN --sweep
```

Runs the hyperparameter grid (depth, width, 50-d vs 300-d GloVe, and dropout), then one-at-a-time follow-ups on the best setting. The follow-ups include tanh, dropout placement, Xavier initialization, Adam at 0.0001, SGD at 0.1, and randomly initialized embeddings (Part 1b). Results are written to `dan_sweep_results.csv`.

## Part 2: Subword DAN using Byte Pair Encodings

```bash
python main.py --model SUBWORDDAN
```

Learns BPE on `data/train.txt` and trains the same network as the Part 1 random-embedding model at vocabulary sizes 100, 200, 500, 1000, 2000, 5000, and 10000. Embeddings are initialized randomly. Prints the dev accuracy for each size, and writes `subword_sweep_results.csv` and `part2_vocab_curve.png`. 
