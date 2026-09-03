"""Spatial multinomial fate model: (phi_i, phi_bar_i) -> P(divide / die / unchanged)."""

import warnings

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

    The interaction term `phi^T B^(k) phi_bar` is controlled by
    `interaction_rank`:

      * a positive int `r` (default 2) factorizes `B^(k) = U^(k) V^(k)^T`
        with `U^(k), V^(k)` of shape (phi_dim, r), so the term becomes
        `sum_j (u_j.phi)(v_j.phi_bar)` — a sum of r "cross-talk modes"
        between own and neighbor state. This is the default because a full
        B's phi_dim*phi_dim outer-product coefficients are wildly collinear
        (see inference.py): the low-rank form uses ~2*phi_dim*r parameters
        instead of phi_dim**2, generalizes better, and is far easier to
        optimize. Its downside is that it is no longer linear in its
        parameters, so per-coefficient Wald inference (fate_parameter_stats)
        does not apply — use likelihood_ratio_test / interaction_modes.
      * `None` keeps the full dense `B^(k)` (linear in parameters, so
        fate_parameter_stats works, but ill-conditioned — fit with a large
        n and expect a near-singular information matrix).
      * `0` drops the interaction entirely (own + neighbor state only). This
        is the model to fit for clean Wald hypothesis testing.

    `n_covariates` adds a per-cell technical-covariate block `d^(k).z_i` to
    each class logit (e.g. log library size, local density; see
    `covariates.py`). These are confounder controls, so — unlike a/c/B — they
    are excluded from weight decay (we want them estimated unbiased, not
    shrunk). When `n_covariates > 0`, `forward`/`fit`/`predict` require a
    matching `covariates` array.
    """

    def __init__(self, phi_dim, class_names=("divide", "die", "unchanged"),
                 interaction_rank=2, n_covariates=0, use_interaction=None):
        super().__init__()
        if len(class_names) < 2:
            raise ValueError("class_names must have at least 2 entries (>=1 free class + 1 reference class)")

        if use_interaction is not None:
            warnings.warn(
                "use_interaction is deprecated; use interaction_rank (None=full dense B, "
                "0=no interaction, positive int=low-rank). Mapping use_interaction="
                f"{use_interaction} to interaction_rank={None if use_interaction else 0}.",
                DeprecationWarning, stacklevel=2,
            )
            interaction_rank = None if use_interaction else 0

        if interaction_rank is not None and (not isinstance(interaction_rank, int) or interaction_rank < 0):
            raise ValueError(f"interaction_rank must be None, 0, or a positive int; got {interaction_rank!r}")

        if not isinstance(n_covariates, int) or n_covariates < 0:
            raise ValueError(f"n_covariates must be a non-negative int; got {n_covariates!r}")

        self.phi_dim = phi_dim
        self.class_names = tuple(class_names)
        self.interaction_rank = interaction_rank
        self.n_covariates = n_covariates
        n_free = len(class_names) - 1

        self.b0 = nn.Parameter(torch.zeros(n_free))
        self.a = nn.Parameter(torch.zeros(n_free, phi_dim))
        self.c = nn.Parameter(torch.zeros(n_free, phi_dim))
        if n_covariates > 0:
            self.d = nn.Parameter(torch.zeros(n_free, n_covariates))

        if interaction_rank is None:
            self.B = nn.Parameter(torch.zeros(n_free, phi_dim, phi_dim))
        elif interaction_rank > 0:
            # Small random init (a zero init would strand U and V at a saddle:
            # dL/dU vanishes when V=0 and vice versa). Seeded via a local
            # generator so construction is reproducible without touching the
            # global RNG state.
            gen = torch.Generator().manual_seed(0)
            scale = 0.01
            self.U = nn.Parameter(torch.randn(n_free, phi_dim, interaction_rank, generator=gen) * scale)
            self.V = nn.Parameter(torch.randn(n_free, phi_dim, interaction_rank, generator=gen) * scale)

    @property
    def interaction_mode(self):
        """'none', 'full', or 'lowrank' — the kind of interaction term in use."""
        if self.interaction_rank is None:
            return "full"
        if self.interaction_rank == 0:
            return "none"
        return "lowrank"

    def forward(self, phi, phi_bar, covariates=None):
        eta_free = self.b0 + phi @ self.a.T + phi_bar @ self.c.T
        if self.interaction_mode == "full":
            eta_free = eta_free + torch.einsum("nd,kde,ne->nk", phi, self.B, phi_bar)
        elif self.interaction_mode == "lowrank":
            own_proj = torch.einsum("nd,kdr->nkr", phi, self.U)
            neighbor_proj = torch.einsum("ne,ker->nkr", phi_bar, self.V)
            eta_free = eta_free + (own_proj * neighbor_proj).sum(dim=-1)
        if self.n_covariates > 0:
            if covariates is None:
                raise ValueError(f"model has n_covariates={self.n_covariates} but no covariates were passed")
            eta_free = eta_free + covariates @ self.d.T
        elif covariates is not None:
            raise ValueError("covariates passed but model was built with n_covariates=0")
        eta_ref = torch.zeros(phi.shape[0], 1, dtype=phi.dtype, device=phi.device)
        return torch.cat([eta_free, eta_ref], dim=1)

    def regularized_parameters(self):
        """Parameters that should receive weight decay (not the intercept b0 or
        the confounder-control covariates d)."""
        params = [self.a, self.c]
        if self.interaction_mode == "full":
            params.append(self.B)
        elif self.interaction_mode == "lowrank":
            params += [self.U, self.V]
        return params

    def unregularized_parameters(self):
        """Parameters excluded from weight decay: the intercept and covariates."""
        params = [self.b0]
        if self.n_covariates > 0:
            params.append(self.d)
        return params

    def interaction_matrix(self):
        """The effective per-class interaction matrix B^(k) (n_free, D, D).

        Returns `B` for a full model and `U V^T` for a low-rank one, so the
        interaction can be inspected / decomposed uniformly regardless of
        parameterization. Raises if the model has no interaction term.
        """
        if self.interaction_mode == "none":
            raise ValueError("model has no interaction term (interaction_rank=0)")
        if self.interaction_mode == "full":
            return self.B.detach().clone()
        return (self.U @ self.V.transpose(-1, -2)).detach()


def fate_loss(logits, target, weight=None):
    """Cross-entropy against hard class indices or soft target distributions.

    `target` is either a 1D LongTensor of class indices (hard labels) or a
    2D float tensor of per-class probabilities summing to 1 (soft targets) —
    dispatched on dtype/shape so callers use one API regardless of which
    kind of target they're training against.

    `weight`, if given, is a 1D tensor of per-class weights (length
    n_classes) that up-weights each class's contribution to the loss —
    the standard remedy for class imbalance, where an unweighted loss lets
    a rare class (e.g. "die" here) be ignored because getting the dominant
    class right already minimizes the average. For hard targets this is
    exactly `F.cross_entropy`'s `weight`; for soft targets it generalizes
    to the same weighted-mean reduction (numerator and denominator both
    weighted by class), so one-hot soft targets reproduce the hard-target
    result.
    """
    if target.ndim == 1 and not torch.is_floating_point(target):
        return F.cross_entropy(logits, target, weight=weight)
    if target.ndim == 2 and torch.is_floating_point(target):
        log_p = F.log_softmax(logits, dim=1)
        if weight is None:
            return -(target * log_p).sum(dim=1).mean()
        weighted_target = target * weight.to(dtype=log_p.dtype)
        numerator = -(weighted_target * log_p).sum()
        denominator = weighted_target.sum().clamp_min(1e-12)
        return numerator / denominator
    raise TypeError("target must be a 1D integer tensor of class indices or a 2D float probability tensor")


def _resolve_class_weight(class_weight, target_t, n_classes):
    """Turn a `class_weight` argument into a per-class float tensor (or None).

    Accepts None (no weighting), an explicit length-`n_classes` array, or
    the string "balanced" — the latter computes inverse-frequency weights
    (`n_samples / (n_classes * class_count)`, sklearn's convention) from
    `target_t`, using hard-label counts for 1D targets and summed target
    mass per class for 2D soft targets.
    """
    if class_weight is None:
        return None

    if isinstance(class_weight, str):
        if class_weight != "balanced":
            raise ValueError(f"Unknown class_weight {class_weight!r}; expected 'balanced', an array, or None")
        if target_t.ndim == 1:
            counts = torch.bincount(target_t, minlength=n_classes).to(torch.float32)
        else:
            counts = target_t.sum(dim=0).to(torch.float32)
        # Guard division only against a class wholly absent here (count 0);
        # its weight is then finite-but-large yet harmless, since no sample
        # contributes that class to the loss. Don't clamp up genuine
        # fractional soft-target mass.
        counts = counts.clamp_min(1e-12)
        n_samples = float(target_t.shape[0])
        return n_samples / (n_classes * counts)

    weight = torch.as_tensor(np.asarray(class_weight), dtype=torch.float32)
    if weight.shape != (n_classes,):
        raise ValueError(f"class_weight array must have shape ({n_classes},); got {tuple(weight.shape)}")
    return weight


def _to_target_tensor(target):
    arr = np.asarray(target)
    if arr.ndim == 1:
        return torch.as_tensor(arr, dtype=torch.long)
    return torch.as_tensor(arr, dtype=torch.float32)


def fit_fate_model(model, phi, phi_bar, target, *, covariates=None, lr=1e-2, weight_decay=1e-3,
                     n_epochs=200, batch_size=None, class_weight=None, verbose=False):
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
    covariates : array-like (n, n_covariates), optional
        Per-cell technical covariates (e.g. `adata.obsm["covariates"]`).
        Required iff the model was built with `n_covariates > 0`.
    batch_size : int, optional
        If given, trains with minibatches of this size each epoch;
        otherwise trains full-batch (fine given how few parameters this
        model has for realistic phi dimensions).
    class_weight : "balanced", array-like, or None
        Per-class loss weights to counter class imbalance. None (default)
        weights every class equally. "balanced" uses inverse-frequency
        weights computed from `target`. An array (length n_classes, in
        `model.class_names` order) sets weights explicitly. Weights are
        computed once from the full `target` and held fixed across
        minibatches, so a rare class is up-weighted consistently regardless
        of how it happens to land in each batch.

    Returns
    -------
    List of per-epoch mean training loss. NB: when `class_weight` is set
    these are weighted-mean losses, so they are not comparable in absolute
    terms to an unweighted run's — compare within a run (that it decreases),
    not across runs with different weighting.
    """
    phi_t = torch.as_tensor(np.asarray(phi), dtype=torch.float32)
    phi_bar_t = torch.as_tensor(np.asarray(phi_bar), dtype=torch.float32)
    cov_t = None if covariates is None else torch.as_tensor(np.asarray(covariates), dtype=torch.float32)
    target_t = _to_target_tensor(target)
    weight_t = _resolve_class_weight(class_weight, target_t, len(model.class_names))

    optimizer = torch.optim.Adam([
        {"params": model.regularized_parameters(), "weight_decay": weight_decay},
        {"params": model.unregularized_parameters(), "weight_decay": 0.0},
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
            cov_batch = None if cov_t is None else cov_t[batch_idx]
            logits = model(phi_t[batch_idx], phi_bar_t[batch_idx], covariates=cov_batch)
            loss = fate_loss(logits, target_t[batch_idx], weight=weight_t)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        history.append(mean_loss)
        if verbose and (epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1):
            print(f"epoch {epoch:4d}  loss {mean_loss:.4f}")

    return history


def predict_fate_proba(model, phi, phi_bar, covariates=None):
    """Predict per-cell class probabilities. Returns an (n, n_classes) numpy array."""
    phi_t = torch.as_tensor(np.asarray(phi), dtype=torch.float32)
    phi_bar_t = torch.as_tensor(np.asarray(phi_bar), dtype=torch.float32)
    cov_t = None if covariates is None else torch.as_tensor(np.asarray(covariates), dtype=torch.float32)
    with torch.no_grad():
        logits = model(phi_t, phi_bar_t, covariates=cov_t)
        proba = F.softmax(logits, dim=1)
    return proba.numpy()
