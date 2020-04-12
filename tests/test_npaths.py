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


class TestCirculator(unittest.TestCase):

    def test_sparam(self):

        node = Circulator(
            freqs=freqs,
            freq_mod=1*GHz,
            C=1*pF)

        S11 = node.compute_sparam(1, 1)
        S21 = node.compute_sparam(2, 1)
        S12 = node.compute_sparam(1, 2)
        S31 = node.compute_sparam(2, 1)

        plt.figure()
        plt.plot(freqs/GHz, 10*np.log10(np.abs(S11)))
        plt.plot(freqs/GHz, 10*np.log10(np.abs(S21)))
        plt.plot(freqs/GHz, 10*np.log10(np.abs(S12)))
        plt.plot(freqs/GHz, 10*np.log10(np.abs(S31)))
        plt.grid()
        plt.show()


if __name__ == '__main__':
    unittest.main()
