"""Spatial multinomial fate model: (phi_i, phi_bar_i) -> P(divide / die / unchanged)."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class SpatialMultinomialFateModel(nn.Module):
    """Multinomial logit model for cell fate with an own-state x neighbor-state interaction.

    For each non-reference class k:
        eta_i^(k) = b0^(k) + a^(k).phi_i + c^(k).phi_bar_i + phi_i^T B^(k) phi_bar_i

    The last entry of `class_names` is the reference class, fixed at logit 0
    — the standard multinomial-logit identifiability constraint (as in
    statsmodels `MNLogit`), rather than a full-K softmax regularized by
    weight decay (which would couple regularization strength to
    identifiability).
    """

    def __init__(self, phi_dim, class_names=("divide", "die", "unchanged")):
        super().__init__()
        if len(class_names) < 2:
            raise ValueError("class_names must have at least 2 entries (>=1 free class + 1 reference class)")
        self.phi_dim = phi_dim
        self.class_names = tuple(class_names)
        n_free = len(class_names) - 1

        self.b0 = nn.Parameter(torch.zeros(n_free))
        self.a = nn.Parameter(torch.zeros(n_free, phi_dim))
        self.c = nn.Parameter(torch.zeros(n_free, phi_dim))
        self.B = nn.Parameter(torch.zeros(n_free, phi_dim, phi_dim))

    def forward(self, phi, phi_bar):
        own = phi @ self.a.T
        neighbor = phi_bar @ self.c.T
        interaction = torch.einsum("nd,kde,ne->nk", phi, self.B, phi_bar)
        eta_free = self.b0 + own + neighbor + interaction
        eta_ref = torch.zeros(phi.shape[0], 1, dtype=phi.dtype, device=phi.device)
        return torch.cat([eta_free, eta_ref], dim=1)

    def regularized_parameters(self):
        """Parameters that should receive weight decay (a, c, B — not b0)."""
        return [self.a, self.c, self.B]


def fate_loss(logits, target):
    """Cross-entropy against hard class indices or soft target distributions.

    `target` is either a 1D LongTensor of class indices (hard labels) or a
    2D float tensor of per-class probabilities summing to 1 (soft targets) —
    dispatched on dtype/shape so callers use one API regardless of which
    kind of target they're training against.
    """
    if target.ndim == 1 and not torch.is_floating_point(target):
        return F.cross_entropy(logits, target)
    if target.ndim == 2 and torch.is_floating_point(target):
        return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    raise TypeError("target must be a 1D integer tensor of class indices or a 2D float probability tensor")


def _to_target_tensor(target):
    arr = np.asarray(target)
    if arr.ndim == 1:
        return torch.as_tensor(arr, dtype=torch.long)
    return torch.as_tensor(arr, dtype=torch.float32)


def fit_fate_model(model, phi, phi_bar, target, *, lr=1e-2, weight_decay=1e-3,
                     n_epochs=200, batch_size=None, verbose=False):
    """Fit a SpatialMultinomialFateModel by full-batch or minibatch Adam.

    `phi`/`phi_bar` are precomputed once from the full neighborhood graph
    (see `neighbors.build_adjacency`/`average_neighbors`), so each row is a
    fixed, independent training example given its target — plain minibatch
    SGD over rows is valid here; no graph-batching machinery is needed since
    nothing is re-aggregated per forward pass.

    Parameters
    ----------
    target : array-like
        Either 1D integer class codes (hard labels, e.g.
        `adata.obs["fate_code"]`) or a 2D array of per-class probabilities
        (soft targets, e.g. `adata.obsm["fate_target"]`).
    batch_size : int, optional
        If given, trains with minibatches of this size each epoch;
        otherwise trains full-batch (fine given how few parameters this
        model has for realistic phi dimensions).

    Returns
    -------
    List of per-epoch mean training loss.
    """
    phi_t = torch.as_tensor(np.asarray(phi), dtype=torch.float32)
    phi_bar_t = torch.as_tensor(np.asarray(phi_bar), dtype=torch.float32)
    target_t = _to_target_tensor(target)

    optimizer = torch.optim.Adam([
        {"params": model.regularized_parameters(), "weight_decay": weight_decay},
        {"params": [model.b0], "weight_decay": 0.0},
    ], lr=lr)

    n = phi_t.shape[0]
    history = []
    for epoch in range(n_epochs):
        if batch_size is None:
            batches = [slice(None)]
        else:
            idx = torch.randperm(n)
            batches = [idx[i:i + batch_size] for i in range(0, n, batch_size)]

        epoch_losses = []
        for batch_idx in batches:
            optimizer.zero_grad()
            logits = model(phi_t[batch_idx], phi_bar_t[batch_idx])
            loss = fate_loss(logits, target_t[batch_idx])
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        history.append(mean_loss)
        if verbose and (epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1):
            print(f"epoch {epoch:4d}  loss {mean_loss:.4f}")

    return history


def predict_fate_proba(model, phi, phi_bar):
    """Predict per-cell class probabilities. Returns an (n, n_classes) numpy array."""
    phi_t = torch.as_tensor(np.asarray(phi), dtype=torch.float32)
    phi_bar_t = torch.as_tensor(np.asarray(phi_bar), dtype=torch.float32)
    with torch.no_grad():
        logits = model(phi_t, phi_bar_t)
        proba = F.softmax(logits, dim=1)
    return proba.numpy()
