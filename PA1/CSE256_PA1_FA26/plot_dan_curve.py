"""Regenerates the Part 1 learning-curve figure for the 8-epoch final model.
Put this next to DANmodels.py / main.py and run:  python plot_dan_curve.py
Writes dan_curve_8ep.png (the report picks it up automatically)."""
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from DANmodels import (FINAL_CONFIG, SEED, build_model, collate_batch,
                       make_optimizer, prepare_data, set_seed, train_dan)

prepared = prepare_data()
word_embeddings, train_data, dev_data = prepared[FINAL_CONFIG["embeddings"]]
set_seed(SEED)
train_loader = DataLoader(train_data, batch_size=FINAL_CONFIG["batch_size"],
                          shuffle=True, collate_fn=collate_batch)
dev_loader = DataLoader(dev_data, batch_size=FINAL_CONFIG["batch_size"],
                        shuffle=False, collate_fn=collate_batch)
model = build_model(word_embeddings, FINAL_CONFIG)
opt = make_optimizer(model, FINAL_CONFIG)
best, hist = train_dan(model, train_loader, dev_loader, opt, FINAL_CONFIG["epochs"], log_every=1)

ep = range(1, len(hist) + 1)
tr_acc, tr_loss, dv_acc, dv_loss = zip(*hist)
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.0))
ax[0].plot(ep, [a * 100 for a in tr_acc], marker="o", label="train")
ax[0].plot(ep, [a * 100 for a in dv_acc], marker="o", label="dev")
ax[0].axvline(best["epoch"], ls="--", c="gray", lw=1)
ax[0].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy"); ax[0].legend(frameon=False)
ax[1].plot(ep, tr_loss, marker="o", label="train")
ax[1].plot(ep, dv_loss, marker="o", label="dev")
ax[1].axvline(best["epoch"], ls="--", c="gray", lw=1)
ax[1].set(xlabel="Epoch", ylabel="Loss", title="Loss"); ax[1].legend(frameon=False)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("dan_curve_8ep.png", dpi=200)
