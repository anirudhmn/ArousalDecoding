"""Decoder training, cross-validation and continuous arousal inference."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from .models import (LogitsOnly, MultiModalClassifierSimple, MultiModalDataset)
from .signals import asymmetric_iir, normalize_train_val

# Training recipe, shared by the offline and online pipelines.
MAX_EPOCHS = 25
PATIENCE = 10
PATIENCE_MIN_EPOCH = 15
BATCH_SIZE = 16
LR = 1e-3
LABEL_SMOOTHING = 0.05
SCHED_STEP, SCHED_GAMMA = 10, 0.1


def get_device(prefer_gpu=True):
    """CUDA if present, else CPU."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_epoch(model, dataloader, optimizer, criterion, scheduler, device):
    model.train()
    total = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x)["logits"], y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    scheduler.step()
    return total / len(dataloader)


@torch.no_grad()
def predict_probs(model, X, device, batch_size=32):
    """Class probabilities for a tensor of epochs, in eval mode."""
    model.eval()
    out = torch.zeros(X.size(0), 2)
    for i in range(0, X.size(0), batch_size):
        logits = model(X[i:i + batch_size].to(device))["logits"]
        out[i:i + batch_size] = F.softmax(logits, dim=1).cpu()
    probs = out.numpy()
    return probs, probs.argmax(axis=1)


def fit_decoder(X_train, y_train, modalities, device, seed=None):
    """Train one decoder to convergence and return it.

    Early stopping tracks *training* loss, which is what the original pipeline
    did: the validation fold is never used for model selection.
    """
    if seed is not None:
        torch.manual_seed(seed)

    model = MultiModalClassifierSimple(modalities, d_model=64, mlp_hidden=128,
                                       num_classes=2, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, SCHED_STEP, SCHED_GAMMA)

    loader = DataLoader(MultiModalDataset(X_train, y_train, augment=True),
                        batch_size=BATCH_SIZE, shuffle=True)

    best_loss, patience = float("inf"), 0
    for epoch in range(1, MAX_EPOCHS + 1):
        if patience >= PATIENCE and epoch > PATIENCE_MIN_EPOCH:
            break
        loss = train_epoch(model, loader, optimizer, criterion, scheduler, device)
        if loss < best_loss:
            best_loss, patience = loss, 0
        else:
            patience += 1
    return model


def integrated_gradients(model, X_val, y_val, n_steps=50, batch_size=8):
    """Mean Integrated Gradients attribution over a validation fold, (C, T)."""
    ig = IntegratedGradients(LogitsOnly(model).eval())
    attrs = []
    for i in range(0, len(X_val), batch_size):
        batch = X_val[i:i + batch_size].clone().requires_grad_(True)
        attr = ig.attribute(batch, target=y_val[i:i + batch_size],
                            n_steps=n_steps,
                            internal_batch_size=min(4, len(batch)))
        attrs.append(attr.detach().cpu().numpy())

    return np.concatenate(attrs, axis=0).mean(axis=0)


def within_subject_cv(ring_df, modalities, device,
                      n_repetitions=10, n_folds=5, n_windows=16,
                      seed_base=0, verbose=True):
    """Repeated stratified k-fold CV within each subject.

    ``ring_df['data']`` must already hold only the channels this configuration
    uses; see ``data.prepare_features``.

    Returns ``(final_results, imps)`` where ``final_results`` is
    (n_subjects, n_repetitions, n_windows, 2) holding accuracy and AUC at each
    of ``n_windows`` cumulative epoch lengths, and ``imps`` is the summed
    Integrated Gradients attribution over every fold.
    """
    subjects = np.unique(ring_df["subj_idx"])
    final_results = np.zeros((len(subjects), n_repetitions, n_windows, 2))
    imps = None

    for sidx, s in enumerate(subjects):
        subj = ring_df[(ring_df["subj_idx"] == s) & (ring_df["label"] >= 0)]
        X = np.stack(subj["data"].to_numpy())
        y = subj["label"].to_numpy()
        n_timesteps = X.shape[-1]
        step = n_timesteps // n_windows
        if imps is None:
            imps = np.zeros((X.shape[1], n_timesteps))

        # Splits are fixed across repetitions; repetitions vary the model only.
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=13)
        splits = list(kf.split(X, y))

        t_subj = time.time()
        for rep in range(n_repetitions):
            y_pred = np.zeros((n_windows, len(y)))
            y_proba = np.zeros((n_windows, len(y), 2))

            for fold, (tr, va) in enumerate(splits):
                Xtr, Xva = normalize_train_val(X[tr], X[va])
                Xtr = torch.tensor(Xtr, dtype=torch.float32).to(device)
                ytr = torch.tensor(y[tr], dtype=torch.long).to(device)
                Xva = torch.tensor(Xva, dtype=torch.float32).to(device)
                yva = torch.tensor(y[va], dtype=torch.long).to(device)

                model = fit_decoder(Xtr, ytr, modalities, device,
                                    seed=seed_base + 1000 * sidx + 10 * rep + fold)

                for w in range(n_windows):
                    probs, preds = predict_probs(
                        model, Xva[:, :, :(w + 1) * step], device)
                    y_pred[w, va] = preds
                    y_proba[w, va] = probs

                imps += integrated_gradients(model, Xva, yva)

            for w in range(n_windows):
                final_results[sidx, rep, w, 0] = accuracy_score(y, y_pred[w])
                final_results[sidx, rep, w, 1] = roc_auc_score(y, y_proba[w, :, 1])

            if verbose:
                print(f"    S{s:02d} rep {rep+1}/{n_repetitions}: "
                      f"AUC {final_results[sidx, rep, -1, 1]*100:5.1f}  "
                      f"[{time.time()-t_subj:.0f}s]", flush=True)

        if verbose:
            auc = final_results[sidx, :, -1, 1]
            print(f"  subject {s:>2}: AUC {auc.mean()*100:5.1f} +/- {auc.std()*100:4.1f}",
                  flush=True)

    return final_results, imps


# --------------------------------------------------------------------------- #
# Continuous arousal decoding
# --------------------------------------------------------------------------- #

@torch.no_grad()
def continuous_arousal(model, trial, lower, upper, device,
                       update=16, window_length=512, batch_size=512):
    """Slide the decoder across one concatenated trial and smooth its output.

    ``trial`` is (C, T). The decoder is re-evaluated every ``update`` samples on
    the most recent ``window_length`` samples; the resulting probability is
    rescaled to 0-100 against the training-set 5th/95th percentiles and passed
    through the asymmetric IIR filter.

    The first ``window_length // 2`` samples are left at zero: no window is
    available yet. This matches the original online implementation, where the
    index sweep starts at sample 256.
    """
    model.eval()
    T = trial.shape[-1]
    arousal = np.zeros(T)
    if T <= 256:
        return arousal

    starts = np.arange(256, T, update)

    # Batch the forward passes by chunk length. The model is in eval mode, so
    # BatchNorm uses running statistics and batching is exactly equivalent to
    # the original one-window-at-a-time loop.
    x = torch.as_tensor(trial, dtype=torch.float32, device=device)
    probs = np.empty(len(starts))
    lengths = np.minimum(starts, window_length)
    for L in np.unique(lengths):
        sel = np.where(lengths == L)[0]
        for b in range(0, len(sel), batch_size):
            idx = sel[b:b + batch_size]
            chunk = torch.stack([x[:, i - L:i] for i in starts[idx]])
            logits = model(chunk)["logits"]
            probs[idx] = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

    # Asymmetric smoothing, then hold each value until the next update.
    alpha_up, alpha_down = asymmetric_iir(update)
    S = 0.0
    for j, i in enumerate(starts):
        A = float(np.clip((probs[j] - lower) / (upper - lower), 0, 1) * 100)
        alpha = alpha_up if A > S else alpha_down
        S = (1 - alpha) * S + alpha * A
        arousal[i:min(i + update, T)] = S
    return arousal


def decode_subject_trials(X_train, y_train, trials, modalities, device,
                          n_repetitions=10, seed_base=0):
    """Train ``n_repetitions`` decoders and decode every trial with each.

    Returns a list, one entry per trial, of (n_repetitions, T) arousal traces.
    """
    out = [np.zeros((n_repetitions, t.shape[-1])) for t in trials]

    Xtr = torch.tensor(X_train, dtype=torch.float32).to(device)
    ytr = torch.tensor(y_train, dtype=torch.long).to(device)

    for rep in range(n_repetitions):
        model = fit_decoder(Xtr, ytr, modalities, device, seed=seed_base + rep)
        probs, _ = predict_probs(model, Xtr, device)
        upper, lower = np.percentile(probs[:, 1], 95), np.percentile(probs[:, 1], 5)
        for t, trial in enumerate(trials):
            out[t][rep] = continuous_arousal(model, trial, lower, upper, device)
    return out
