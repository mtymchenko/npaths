import numpy as np
from scipy.signal import square
from scipy.linalg import toeplitz, inv


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
        self.T_mod = 1/self.freq_mod
        self.tau = self.C * self.Z0
        self.harmonics = np.arange(-self.n_harmonics, self.n_harmonics+1)

        self._smatrix = []
        for _ in range(self.n_ports):
            row = []
            for _ in range(self.n_ports):
                row.append(None)
            self._smatrix.append(row)

        N = self.n_harmonics

        self.U = np.zeros((2*N+1, 2*N+1), dtype=complex)

        pt = [None] * self.n_ports
        pn = [None] * self.n_ports
        self.P = [None] * self.n_ports

        t = np.linspace(-self.T_mod/2, self.T_mod/2, 2**12)

        for iport in range(self.n_ports):

            pt[iport] = 0.5*(1+square(
                2*np.pi*(t/self.T_mod-self.delays[iport]),
                self.duty_cycles[iport]))

            pn[iport] = get_spectrum(pt[iport])

            pn_p = pn[iport][np.arange(0, 2*N+1, 1)]
            pn_m = pn[iport][np.arange(0, -2*N-1, -1)]

            self.P[iport] = toeplitz(pn_p, pn_m)
            self.U += self.P[iport]

    def sparam(self, port_to, port_from, harmonic=0):
        """Computes desired sparams.

        Args:
            port_to (int): Output port.
            port_from (int): Input port.
            harmonic (int, optional): S-param harmonic.

        Returns:
            sparam (ndarray): Computed s-params.

        """
        p1 = port_to-1
        p2 = port_from-1

        if self._smatrix[p1][p2] is None:

            N = self.n_harmonics
            N1 = self.n_harmonics_subset
            harmonics = self.harmonics
            tau = self.tau
            freq_mod = self.freq_mod
            U = self.U
            P = self.P[port_from-1]
            Q = self.P[port_to-1]
            offdiagU = U - np.diag(np.diag(U))

            sparam = np.zeros((2*N+1, len(self.freqs)), dtype=complex)

            for ifreq, freq in enumerate(self.freqs):

                r0 = -np.around(freq/freq_mod)
                rr = np.arange(r0-N1, r0+N1+1)
                ll = np.setxor1d(harmonics, rr)

                l1 = (ll+N).astype(np.intp)
                l1v = l1[:, np.newaxis]

                r1 = (rr+N).astype(np.intp)
                r1v = r1[:, np.newaxis]

                Om = 2*np.pi*np.diag(freq + harmonics*freq_mod)

                M = U + 1j*Om*tau

                B = np.diag(np.diag(M))
                invBl1 = inv(B[l1v, l1])
                # invMr1 = inv(M[r1v, r1])

                F = np.matmul(offdiagU[:, l1], invBl1)

                P1 = P - np.matmul(F, P[l1, :])
                U1 = U - np.matmul(F, U[l1, :])

                M1 = U1 + 1j*Om*tau
                invM1r1 = inv(M1[r1v, r1])

                vc = np.zeros((2*N+1, 1), dtype=complex)
                vc[r1] = np.matmul(invM1r1, P1[r1v, N])
                vc[l1] = np.matmul(
                    invBl1, P1[l1v, N] - U1[l1v, r1].dot(vc[r1]))

                sN = harmonics[np.where(np.mod(harmonics, self.n_paths) == 0)]
                sN1 = (sN+N).astype(np.intp)

                if port_from == port_to:
                    sparam[sN, ifreq] = np.squeeze(
                        2 * self.n_paths * Q[sN1, :].dot(vc) - 1)
                else:
                    sparam[sN, ifreq] = np.squeeze(
                        2 * self.n_paths * Q[sN1, :].dot(vc))

            self._smatrix[p1][p2] = sparam

        return self._smatrix[p1][p2][harmonic, :]


def get_spectrum(func):
    spectrum = np.fft.fft(np.real(func))/len(func)
    return spectrum
