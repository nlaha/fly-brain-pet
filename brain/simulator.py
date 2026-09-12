"""
Leaky-rate simulation over the connectome's sparse adjacency matrix.
r' = (-r + relu(W @ r + I_ext)) / tau

Sparse mm is used every step so this scales to the full ~166k-neuron
graph on GPU; on CPU it'll still run, just slower per step (use
build_curated_subgraph for a CPU-friendly size).
"""
import torch

from .connectome import Connectome


class LeakyRateSimulator:
    def __init__(self, connectome: Connectome, tau: float = 10.0, dt: float = 1.0, device: str = "cpu"):
        self.c = connectome
        self.device = device
        self.tau = tau
        self.dt = dt
        # One persistent state vector per brain.  The connectome graph itself is
        # shared by all flies; only this state is unique to each fly.
        self.rates = torch.zeros(connectome.n_neurons, device=device, dtype=torch.float32)
        self.external_input = torch.zeros_like(self.rates)
        self._scratch = torch.empty_like(self.rates)
        # Do not call .to() per fly: the Connectome already owns the device copy.
        self.adjacency = connectome.adjacency

    def step(self, external_input: torch.Tensor) -> torch.Tensor:
        """external_input: dense [n_neurons] tensor of injected drive (zeros
        for neurons with no stimulus this step)."""
        # Reuse buffers and disable autograd. This is inference, not training,
        # so keeping an autograd graph would waste GPU memory every frame.
        with torch.inference_mode():
            synaptic = torch.sparse.mm(self.adjacency, self.rates.unsqueeze(1)).squeeze(1)
            self._scratch.copy_(synaptic)
            self._scratch.add_(external_input)
            self._scratch.relu_()
            self.rates.add_((self.dt / self.tau) * (-self.rates + self._scratch))
            self.rates.clamp_(0.0, 50.0)
        return self.rates

    def reset(self):
        self.rates.zero_()
        self.external_input.zero_()
