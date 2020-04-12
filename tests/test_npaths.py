import unittest
import numpy as np
import matplotlib.pyplot as plt

from npaths import NPathNode, Filter, Circulator


GHz = 1e9
ohm = 1
pF = 1e-12
freqs = np.linspace(0.001, 6, 500)*GHz


class TestNPathNode(unittest.TestCase):

    def test_sparam(self):

        node = NPathNode(
            freqs=freqs,
            freq_mod=1*GHz,
            C=3*pF)

        S11 = node.compute_sparam(1, 1)
        S21 = node.compute_sparam(2, 1)

        plt.figure()
        plt.plot(freqs/GHz, 10*np.log10(np.abs(S11)))
        plt.plot(freqs/GHz, 10*np.log10(np.abs(S21)))
        plt.grid()
        plt.show()

        # plt.figure()
        # plt.plot(freqs/GHz, np.unwrap(np.angle(S11[N, :]))/np.pi)
        # plt.plot(freqs/GHz, np.unwrap(np.angle(S21[N, :]))/np.pi)
        # plt.plot(freqs/GHz, np.unwrap(np.angle(S12[N, :]))/np.pi)
        # plt.grid()
        # plt.show()


class TestFilter(unittest.TestCase):

    def test_sparam(self):

        node = Filter(
            freqs=freqs,
            freq_mod=1*GHz,
            C=15*pF)

        S11 = node.compute_sparam(1, 1)
        S21 = node.compute_sparam(2, 1)

        plt.figure()
        plt.plot(freqs/GHz, 10*np.log10(np.abs(S11)))
        plt.plot(freqs/GHz, 10*np.log10(np.abs(S21)))
        plt.grid()
        plt.show()


if __name__ == '__main__':
    unittest.main()
