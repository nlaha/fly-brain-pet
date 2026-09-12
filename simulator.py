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
        self.rates = torch.zeros(connectome.n_neurons, device=device)
        self.adjacency = connectome.adjacency.to(device)

    def step(self, external_input: torch.Tensor) -> torch.Tensor:
        """external_input: dense [n_neurons] tensor of injected drive (zeros
        for neurons with no stimulus this step)."""
        synaptic = torch.sparse.mm(self.adjacency, self.rates.unsqueeze(1)).squeeze(1)
        drive = torch.relu(synaptic + external_input)
        self.rates = self.rates + (self.dt / self.tau) * (-self.rates + drive)
        self.rates.clamp_(0.0, 50.0)  # cap firing rate, keeps things numerically stable
        return self.rates

    def reset(self):
        self.rates.zero_()
