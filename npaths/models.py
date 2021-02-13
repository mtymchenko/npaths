from typing import Optional, List, Sequence
import numpy as np

from .general import NPath


__all__ = [
    'NPathFilter',
    'Circulator'
]


class NPathFilter(NPath):
    """Analytical model of a 2-port N-path filter."""

    def __init__(self,
                 freqs: List[float],
                 freq_mod: float,
                 C: float,
                 n_paths: int = 8,
                 n_harmonics: int = 40,
                 n_harmonics_subset: int = 7,
                 delays: Optional[List[float]] = None,
                 duty_cycles: Optional[List[float]] = None,
                 Z0: float = 50.0):

        n_ports = 2

        if delays is None:
            delays = np.arange(0.0, 1.0, 1/n_ports)

        if duty_cycles is None:
            duty_cycles = np.ones((n_ports, ))*1.0/n_paths

        super().__init__(
            freqs, freq_mod, C,
            n_ports, n_paths, n_harmonics, n_harmonics_subset,
            delays, duty_cycles, Z0)


class Circulator(NPath):
    """Analytical model of a M-way N-path circulator based on switched
    capacitors.

    """
    def __init__(self,
                 freqs: Sequence[float],
                 freq_mod: float,
                 C: float,
                 n_ports: int = 3,
                 n_paths: int = 9,
                 n_harmonics: int = 40,
                 n_harmonics_subset: int = 7,
                 delays: Optional[List[float]] = None,
                 duty_cycles: Optional[List[float]] = None,
                 Z0: float = 50.0):

        if delays is None:
            delays = np.arange(0.0, 1.0, 1/n_ports)

        if duty_cycles is None:
            duty_cycles = np.ones((n_ports, ))*1.0/n_paths

        super().__init__(
            freqs, freq_mod, C,
            n_ports, n_paths, n_harmonics, n_harmonics_subset,
            delays, duty_cycles, Z0)
