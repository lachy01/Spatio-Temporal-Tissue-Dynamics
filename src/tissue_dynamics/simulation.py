"""Phase-2 stub: time-stepping tissue dynamics driven by the fitted fate model.

Not implemented yet — this module only fixes the interface so the static
pipeline (`pipeline.run_static_pipeline`) has a clear handoff point once
temporal simulation is built.

Two candidate designs to choose between when this is implemented:

1. **Discrete agent-based**: each step, sample a divide/die/unchanged event
   per cell from `predict_fate_proba`; remove dying cells; spawn daughter
   cells near dividing parents; rebuild the spatial adjacency (`neighbors
   .build_adjacency`) since cell positions/counts change every step. Keeps
   per-cell identity and exact positions; cost grows with cell count.

2. **Continuum binned-density**: discretize the tissue into a fixed spatial
   grid, track density (and an averaged phi) per bin, and evolve it with
   reaction-diffusion-style update rules (birth/death sourced from local
   class probabilities, migration as inter-bin flux). Trades cell identity
   for scalability to large tissue areas / long time horizons.
"""

from typing import Protocol

from anndata import AnnData


class DynamicsStep(Protocol):
    """Interface a phase-2 simulation step implementation must satisfy."""

    def step(self, adata: AnnData, dt: float) -> AnnData:
        """Advance tissue state by `dt` and return the updated AnnData."""
        raise NotImplementedError(
            "Temporal simulation is not implemented yet — see module docstring "
            "for the two candidate designs (discrete agent-based vs. continuum "
            "binned-density) to choose between when building this out."
        )
