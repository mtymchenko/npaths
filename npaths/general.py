import numpy as np
from scipy.signal import square
from scipy.linalg import toeplitz, inv

from .helpers import read_str_sparam


__all__ = [
    'NPathNode'
]


class NPathNode:
    """Model of an N-way circulator based on switched capacitive
    networks.

    Args:
        freqs (list of float): Simulation frequencies (Hz).
        freq_mod (float): Switching frequency (Hz).
        C (float): Branch capacitance (F).
        n_ports (int, optional): Number of ports.
        n_paths (int, optional): Number of paths.
        n_harmonics (int, optional): Number of harmonics
            [-n_harmonics...n_harmonics].
        delays (list of float): Delays on switches at all ports.
            Each must be between 0 and 1.
        duty_cycles (list of floats): Duty cycles at all ports.
            Each must be between 0 and 1.

    """
    def __init__(
            self, freqs, freq_mod, C,
            n_ports=2, n_paths=8, n_harmonics=40, n_harmonics_subset=7,
            delays=[0.0, 0.5], duty_cycles=[1.0/8.0, 1.0/8.0],
            Z0=50.0):

        self.freqs = freqs
        self.freq_mod = freq_mod
        self.C = C
        self.n_ports = n_ports
        self.n_paths = n_paths
        self.n_harmonics = n_harmonics
        self.n_harmonics_subset = n_harmonics_subset
        self.delays = delays
        self.duty_cycles = duty_cycles
        self.Z0 = Z0

        N = self.n_harmonics

        self.U = np.zeros((2*N+1, 2*N+1), dtype=complex)

        pt = [None] * self.n_ports
        pn = [None] * self.n_ports
        self.P = [None] * self.n_ports

        t = np.linspace(-self.T_mod/2, self.T_mod/2, 2**12)

        for ip in range(self.n_ports):

            pt[ip] = 0.5*(1+square(
                2*np.pi*(t/self.T_mod-self.delays[ip]), self.duty_cycles[ip]))
            pn[ip] = get_spectrum(pt[ip])

            pn_p = pn[ip][np.arange(0, 2*N+1, 1)]
            pn_m = pn[ip][np.arange(0, -2*N-1, -1)]

            self.P[ip] = toeplitz(pn_p, pn_m)
            self.U += self.P[ip]

    @property
    def omegas(self):
        return 2*np.pi*self.freqs

    @property
    def omega_mod(self):
        return 2*np.pi*self.freq_mod

    @property
    def T_mod(self):
        return 1/self.freq_mod

    @property
    def tau(self):
        return self.C * self.Z0

    @property
    def harmonics(self):
        return np.arange(-self.n_harmonics, self.n_harmonics+1)

    def get_sparams(self, sparams):
        pass

    def compute_sparam(self, port_to, port_from, harmonic=0):
        """Computes desired sparams.

        Args:
            port_to (int): Output port.
            port_from (int): Input port.
            harmonic (int, optional): S-param harmonic.

        Returns:
            sparam (ndarray): Computed s-params.

        """
        N = self.n_harmonics
        N_subset = self.n_harmonics_subset
        harmonics = self.harmonics
        tau = self.tau
        omega_mod = self.omega_mod
        U = self.U

        sparam = np.zeros((2*N+1, len(self.freqs)), dtype=complex)

        for ifreq, _ in enumerate(self.freqs):

            omega = self.omegas[ifreq]
            r0 = -np.around(omega/self.omega_mod)
            r = np.arange(r0-N_subset, r0+N_subset+1)
            l = np.setxor1d(harmonics, r)

            B = np.diag(U[N, N] + 1j * (omega + harmonics*omega_mod) * tau)
            diagU = np.diag(np.diag(U))

            l1 = (l+N).astype(np.intp)
            l1v = l1[:, np.newaxis]

            P = self.P[port_from-1]
            Q = self.P[port_to-1]

            P1 = P - np.linalg.multi_dot(
                [U[:, l1] - diagU[:, l1], inv(B[l1v, l1]), P[l1, :]])

            U1 = U - np.linalg.multi_dot(
                [U[:, l1] - diagU[:, l1], inv(B[l1v, l1]), U[l1, :]])

            r1 = (r+N).astype(np.intp)
            r1v = r1[:, np.newaxis]

            Om = np.diag(omega + r*omega_mod)

            vc = np.zeros((2*N+1, 1), dtype=complex)
            vc[r1] = np.matmul(inv(U1[r1v, r1] + 1j*Om*tau), P1[r1v, N])
            vc[l1] = np.matmul(
                inv(B[l1v, l1]), P1[l1v, N] - np.matmul(U1[l1v, r1], vc[r1]))

            sN = harmonics[np.where(np.mod(harmonics, self.n_paths) == 0)]
            sN1 = (sN+N).astype(np.intp)

            if port_from == port_to:
                sparam[sN, ifreq] = np.squeeze(
                    2 * self.n_paths * np.dot(Q[sN1, :], vc) - 1)
            else:
                sparam[sN, ifreq] = np.squeeze(
                    2 * self.n_paths * np.dot(Q[sN1, :], vc))

        return sparam[harmonic, :]


def get_spectrum(function):
    spectrum = np.fft.fft(np.real(function))/len(function)
    return spectrum


