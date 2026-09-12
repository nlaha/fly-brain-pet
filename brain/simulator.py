"""GPU-efficient batched leaky-rate simulation over the full connectome."""
from __future__ import annotations
import torch
from .connectome import Connectome


class BatchedLeakyRateSimulator:
    """N independent brain states sharing one immutable sparse connectome.

    State is [neurons, flies], so one sparse-dense matmul advances every brain
    together. The graph and weights are not duplicated.
    """
    def __init__(self, connectome: Connectome, batch_size: int, tau: float = 10.0,
                 dt: float = 1.0, device: str = "cpu", intrinsic_noise: float = 0.008):
        self.c = connectome
        self.device = device
        self.tau = tau
        self.dt = dt
        self.batch_size = batch_size
        self.intrinsic_noise = intrinsic_noise
        shape = (connectome.n_neurons, batch_size)
        self.rates = torch.zeros(shape, device=device, dtype=torch.float32)
        self.external_input = torch.zeros_like(self.rates)
        self._scratch = torch.empty_like(self.rates)
        self.adjacency = connectome.adjacency
        # CSR generally uses less index storage than COO and is the preferred
        # format for repeated sparse-matrix-vector/matrix products.
        if device == "cuda" and getattr(self.adjacency, "layout", None) == torch.sparse_coo:
            try:
                self.adjacency = self.adjacency.to_sparse_csr()
            except RuntimeError:
                pass

    @torch.inference_mode()
    def step(self) -> torch.Tensor:
        synaptic = torch.sparse.mm(self.adjacency, self.rates)
        self._scratch.copy_(synaptic)
        # Stochasticity is modeled as rate-dependent synaptic bombardment,
        # rather than injecting a fixed amount of activity into every neuron.
        # This is a rate-model approximation of biological synaptic noise:
        # active neurons get more variance, while silent neurons are not
        # continuously kicked above threshold. It is not a calibrated model
        # of Drosophila noise, so keep it deliberately small.
        if self.intrinsic_noise > 0:
            drive = self._scratch.clamp_min(0.0)
            sigma = self.intrinsic_noise * torch.sqrt(drive + 1e-4)
            self._scratch.add_(torch.randn_like(self._scratch) * sigma)
        self._scratch.add_(self.external_input)
        self._scratch.relu_()
        self.rates.add_((self.dt / self.tau) * (-self.rates + self._scratch))
        self.rates.clamp_(0.0, 50.0)
        return self.rates

    def reset(self):
        self.rates.zero_()
        self.external_input.zero_()
