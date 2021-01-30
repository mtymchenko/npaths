import numpy as np

from .general import NPath


__all__ = [
    'NPathFilter',
    'Circulator'
]


class NPathFilter(NPath):
    """Analytical model of a M-way N-path circulator based on switched
    capacitors.


    """
    def __init__(
            self, freqs, freq_mod, C,
            n_paths=8, n_harmonics=40, n_harmonics_subset=7,
            delays=None, duty_cycles=None,
            Z0=50.0):

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
    def __init__(
            self, freqs, freq_mod, C,
            n_ports=3, n_paths=9, n_harmonics=40, n_harmonics_subset=7,
            delays=None, duty_cycles=None,
            Z0=50.0):

        if delays is None:
            delays = np.arange(0.0, 1.0, 1/n_ports)

        if duty_cycles is None:
            duty_cycles = np.ones((n_ports, ))*1.0/n_paths

        super().__init__(
            freqs, freq_mod, C,
            n_ports, n_paths, n_harmonics, n_harmonics_subset,
            delays, duty_cycles, Z0)
