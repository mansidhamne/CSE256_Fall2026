"""Train Parts 1–2: python main.py --part part1|part2."""
import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from tokenizer import SimpleTokenizer
from transformer import EncoderClassifier, TransformerEncoder, TransformerDecoder
from utilities import Utilities

# Supplied base hyperparameters.
seed = 42
batch_size = 16
block_size = 32
learning_rate = 1e-3
n_embd = 64
n_head = 2
n_layer = 4
n_input = 64
n_hidden = 100
n_output = 3
epochs_CLS = 15
# Decoder training settings.
eval_interval = 100
max_iters = 500
eval_iters = 200

ROOT = Path(__file__).resolve().parent


def load_texts(directory):
    """Build the supplied word vocabulary from training files only."""
    return [p.read_text(encoding='utf-8') for p in sorted(Path(directory).iterdir())
            if p.is_file() and 'test' not in p.name]


def collate_batch(batch):
    data, labels = zip(*batch)
    padded = pad_sequence(data, batch_first=True, padding_value=0)[:, :block_size]
    padded = nn.functional.pad(padded, (0, max(0, block_size - padded.shape[1])))
    return padded, torch.stack(labels)


def cached_dataset(dataset):
    # Tokenize once rather than repeating NLTK sentence splitting each epoch.
    x, y = collate_batch([dataset[i] for i in range(len(dataset))])
    return TensorDataset(x, y)


@torch.no_grad()
def compute_classifier_accuracy(classifier, data_loader, device):
    was_training = classifier.training
    classifier.eval()
    correct = total = 0
    for x, y in data_loader:
        predictions = classifier(x.to(device)).argmax(dim=-1).cpu()
        correct += (predictions == y).sum().item()
        total += y.numel()
    classifier.train(was_training)
    return 100 * correct / total


def train_part1(args):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(args.threads)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    data_dir = ROOT / 'speechesdataset'

    tokenizer = SimpleTokenizer(' '.join(load_texts(data_dir)))
    training = cached_dataset(SpeechesClassificationDataset(tokenizer, data_dir / 'train_CLS.tsv'))
    testing = cached_dataset(SpeechesClassificationDataset(tokenizer, data_dir / 'test_CLS.tsv'))

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(testing, batch_size=batch_size, shuffle=False)

    encoder = TransformerEncoder(tokenizer.vocab_size, block_size, n_embd, n_head, n_layer)

    model = EncoderClassifier(encoder, n_embd, n_hidden, n_output).to(device)

    encoder_parameters = sum(p.numel() for p in encoder.parameters())
    classifier_parameters = sum(p.numel() for p in model.classifier.parameters())

    print(f'Device: {device}; vocabulary: {tokenizer.vocab_size}; train: {len(training)}; test: {len(testing)}', flush=True)
    print(f'Encoder parameters: {encoder_parameters:,}; classifier parameters: {classifier_parameters:,}', flush=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    history = []
    for epoch in range(1, epochs_CLS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.numel()
        accuracy = compute_classifier_accuracy(model, test_loader, device)
        history.append({'epoch': epoch, 'train_loss': total_loss / len(training),
                        'test_accuracy_percent': accuracy})
        print(f'Epoch {epoch:2d}/{epochs_CLS}: loss={history[-1]["train_loss"]:.4f}, test accuracy={accuracy:.2f}%', flush=True)

        with (output / 'metrics.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    model.eval()
    
    torch.save({'model_state_dict': model.cpu().state_dict(), 'stoi': tokenizer.stoi,
                'config': {'vocab_size': tokenizer.vocab_size, 'block_size': block_size,
                           'n_embd': n_embd, 'n_head': n_head, 'n_layer': n_layer,
                           'n_hidden': n_hidden, 'n_output': n_output}, 'seed': seed}, output / 'part1_model.pt')
   
    sentences = ['I will invest in early childhood education.',
                 'We have the power to make the world we seek.']
    
    checks = [Utilities(tokenizer, model.encoder).sanity_check(
        sentence, block_size, output_dir=output, prefix=f'sentence_{i}',
        selected_maps=(0,), n_head=n_head) for i, sentence in enumerate(sentences, 1)]
    
    results = {'seed': seed, 'device': str(device), 'torch_version': torch.__version__,
               'vocabulary_size': tokenizer.vocab_size, 'train_samples': len(training),
               'test_samples': len(testing), 'encoder_parameters': encoder_parameters,
               'classifier_parameters': classifier_parameters,
               'final_test_accuracy_percent': history[-1]['test_accuracy_percent'],
               'history': history, 'attention_checks': checks,
               'hyperparameters': {'batch_size': batch_size, 'block_size': block_size,
                                   'learning_rate': learning_rate, 'n_embd': n_embd,
                                   'n_head': n_head, 'n_layer': n_layer,
                                   'encoder_ffn_hidden': 4 * n_embd,
                                   'classifier_hidden': n_hidden, 'epochs': epochs_CLS}}
    (output / 'results.json').write_text(json.dumps(results, indent=2))
    
    print(f'Results written to {output}', flush=True)


def lm_evaluation_dataset(token_ids):
    """Score every next-token target once, in blocks with reset position/context.

    Unlike the stride-one training dataset, evaluation uses consecutive blocks.
    The final partial block is right-padded and padded targets are ignored.
    """
    xs, ys = [], []
    for start in range(0, len(token_ids) - 1, block_size):
        x = torch.tensor(token_ids[start:start + block_size], dtype=torch.long)
        y = torch.tensor(token_ids[start + 1:start + block_size + 1], dtype=torch.long)
        # At the tail there may be one more input token than target tokens.
        x = x[:len(y)]
        xs.append(nn.functional.pad(x, (0, block_size - len(x)), value=0))
        ys.append(nn.functional.pad(y, (0, block_size - len(y)), value=-100))
    if not xs:
        raise ValueError('Language-model evaluation needs at least two tokens')
    return TensorDataset(torch.stack(xs), torch.stack(ys))


@torch.no_grad()
def compute_perplexity(model, data_loader, device):
    """exp(total negative log likelihood / number of valid target tokens)."""
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        total_nll += nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
            ignore_index=-100, reduction='sum').item()
        total_tokens += y.ne(-100).sum().item()
    model.train(was_training)
    return {'perplexity': math.exp(total_nll / total_tokens),
            'mean_nll': total_nll / total_tokens, 'target_tokens': total_tokens}


def train_part2(args):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(args.threads)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT / 'speechesdataset'

    tokenizer = SimpleTokenizer(' '.join(load_texts(data_dir)))

    train_text = (data_dir / 'train_LM.txt').read_text(encoding='utf-8')
    training = LanguageModelingDataset(tokenizer, train_text, block_size)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=True,
                              drop_last=True, generator=generator)
    evaluation_loaders = {}
    token_stats = {}
    for split, filename in [('train', 'train_LM.txt'), ('obama', 'test_LM_obama.txt'),
                            ('wbush', 'test_LM_wbush.txt'), ('hbush', 'test_LM_hbush.txt')]:
        ids = tokenizer.encode((data_dir / filename).read_text(encoding='utf-8'))
        evaluation_loaders[split] = DataLoader(lm_evaluation_dataset(ids), batch_size=batch_size)
        token_stats[split] = {'tokens': len(ids), 'unknown_tokens': ids.count(1),
                              'unknown_fraction': ids.count(1) / len(ids)}
    model = TransformerDecoder(tokenizer.vocab_size, block_size, n_embd, n_head, n_layer).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f'Device: {device}; vocabulary: {tokenizer.vocab_size}; decoder parameters: {parameters:,}', flush=True)
    print('Evaluation targets:', {k: v['tokens'] - 1 for k, v in token_stats.items()}, flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    detailed_evaluations = []
    train_batches = iter(train_loader)
    recent_loss = 0.0
    for iteration in range(1, max_iters + 1):
        try:
            x, y = next(train_batches)
        except StopIteration:
            train_batches = iter(train_loader)
            x, y = next(train_batches)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = model(x.to(device), y.to(device))
        loss.backward()
        optimizer.step()
        recent_loss += loss.item()
        if iteration % eval_interval == 0 or iteration == max_iters:
            measured = {split: compute_perplexity(model, loader, device)
                        for split, loader in evaluation_loaders.items()}
            row = {'iteration': iteration, 'recent_train_loss': recent_loss / eval_interval}
            row.update({f'{split}_perplexity': v['perplexity'] for split, v in measured.items()})
            history.append(row)
            detailed_evaluations.append({'iteration': iteration, 'splits': measured})
            recent_loss = 0.0
            print(f'Iteration {iteration}/{max_iters}: ' + ', '.join(
                f'{split} PPL={v["perplexity"]:.2f}' for split, v in measured.items()), flush=True)
            with (output / 'metrics.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
    model.eval()
    torch.save({'model_state_dict': model.cpu().state_dict(), 'stoi': tokenizer.stoi,
                'config': {'vocab_size': tokenizer.vocab_size, 'block_size': block_size,
                           'n_embd': n_embd, 'n_head': n_head, 'n_layer': n_layer},
                'seed': seed}, output / 'part2_model.pt')

    sentences = ['I will invest in early childhood education.',
                 'We have the power to make the world we seek.']

    checks = [Utilities(tokenizer, model).sanity_check(
        sentence, block_size, output_dir=output, prefix=f'sentence_{i}',
        selected_maps=(0,), n_head=n_head, causal=True)
        for i, sentence in enumerate(sentences, 1)]
    results = {'seed': seed, 'device': str(device), 'torch_version': torch.__version__,
               'vocabulary_size': tokenizer.vocab_size, 'decoder_parameters': parameters,
               'token_stats': token_stats, 'history': history,
               'evaluations': detailed_evaluations, 'attention_checks': checks,
               'evaluation_protocol': 'All next-token targets once, consecutive length-32 blocks, '
                                      'context/position reset per block; padded targets ignored; '
                                      'perplexity = exp(token-weighted mean cross-entropy).',
               'hyperparameters': {'batch_size': batch_size, 'block_size': block_size,
                                   'learning_rate': learning_rate, 'n_embd': n_embd,
                                   'n_head': n_head, 'n_layer': n_layer,
                                   'decoder_ffn_hidden': 100, 'iterations': max_iters,
                                   'training_token_presentations': max_iters * batch_size * block_size}}
    (output / 'results.json').write_text(json.dumps(results, indent=2))
    print(f'Results written to {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=['part1', 'part2'], default='part1')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = str(ROOT / 'outputs' / args.part)
    (train_part1 if args.part == 'part1' else train_part2)(args)


if __name__ == '__main__':
    main()
