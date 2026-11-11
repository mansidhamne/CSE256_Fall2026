"""Attention sanity checks and token-labelled heatmaps."""
from pathlib import Path
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


def save_attention_plot(attention, tokens, layer, head, causal, path):
    """Plot probabilities and unrounded row sums, and export numeric values."""
    fig, ax = plt.subplots(figsize=(max(9, len(tokens) * .65), max(7, len(tokens) * .55)))
    picture = ax.imshow(attention.numpy(), cmap='viridis', vmin=0, vmax=1)
    for row in range(len(tokens)):
        for column in range(len(tokens)):
            value = attention[row, column].item()
            ax.text(column, row, f'{value:.3f}', ha='center', va='center',
                    fontsize=7, color='black' if value > .55 else 'white')
    ax.set_xticks(range(len(tokens)), tokens, rotation=60, ha='right')
    ax.set_yticks(range(len(tokens)), tokens)
    sums = ax.secondary_yaxis('right')
    sums.set_yticks(range(len(tokens)), [f'{v:.2f}' for v in attention.sum(-1).tolist()])
    sums.set_ylabel('Row sum')
    ax.set_xlabel('Key')
    ax.set_ylabel('Query')
    ax.set_title(f'{"Decoder" if causal else "Encoder"} layer {layer}, head {head}')
    fig.colorbar(picture, ax=ax, label='Attention probability', pad=.22)
    # fig.text(.5, .01, 'Cell values rounded to 3 decimals; rounding can make their displayed sum differ slightly from 1.',
    #          ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    with Path(path).with_suffix('.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Query / Key', *tokens, 'row_sum'])
        for token, row in zip(tokens, attention):
            writer.writerow([token, *row.tolist(), row.sum().item()])


class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    @torch.no_grad()
    def sanity_check(self, sentence, block_size, output_dir='.', prefix='attention',
                     selected_maps=(0,), n_head=2, causal=False):
        wordids = self.tokenizer.encode(sentence)[:block_size]
        if not wordids:
            raise ValueError('Sanity-check sentence must contain tokens')
        device = next(self.model.parameters()).device
        ids = torch.tensor([wordids + [0] * (block_size - len(wordids))], device=device)
        was_training = self.model.training
        self.model.eval()
        _, maps = self.model(ids)
        self.model.train(was_training)
        row_error = max((a.sum(-1) - 1).abs().max().item() for a in maps)
        assert row_error < 1e-5, f'Attention rows are not normalized: {row_error}'
        padding_max = max((a[..., len(wordids):].abs().max().item()
                           if len(wordids) < block_size else 0.0) for a in maps)
        assert padding_max == 0.0, 'Padding keys received attention'
        future_max = max(a.triu(1).abs().max().item() for a in maps)
        if causal:
            assert future_max == 0.0, 'Causal attention leaked to future keys'
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        tokens = [self.tokenizer.itos[i] for i in wordids]
        paths = []
        summaries = []
        for index in selected_maps:
            attention = maps[index][0, :len(wordids), :len(wordids)].cpu()
            layer, head = index // n_head + 1, index % n_head + 1
            path = directory / f'{prefix}_layer{layer}_head{head}.png'
            save_attention_plot(attention, tokens, layer, head, causal, path)
            paths.append(str(path))
            # Save the full numeric maps for reproducible figure inspection.
            torch.save(attention, path.with_suffix('.pt'))
            mean_key_attention = attention.mean(dim=0)
            best = mean_key_attention.argmax().item()
            summaries.append({'layer': layer, 'head': head,
                              'most_attended_key': tokens[best],
                              'most_attended_key_position': best,
                              'mean_attention_to_key': mean_key_attention[best].item(),
                              'future_attention_mass_mean': attention.triu(1).sum().item() / len(tokens)})
        print(f'Attention check: {len(maps)} heads; max row-sum error={row_error:.2e}; '
              f'max padding attention={padding_max:.1f}', flush=True)
        return {'sentence': sentence, 'tokens': tokens, 'maps_checked': len(maps),
                'max_row_sum_error': row_error, 'max_padding_attention': padding_max,
                'max_future_attention': future_max,
                'plots': paths, 'plot_summaries': summaries}
