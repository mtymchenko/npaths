import numpy as np
import matplotlib.pyplot as plt

from npaths import NPathFilter
from npaths.helpers import mag2db


GHz = 1e9
pF = 1e-12

freqs = np.linspace(1e-3, 8.0, 2**8)*GHz

npf = NPathFilter(
    freqs=freqs,
    freq_mod=1*GHz,
    C=10*pF,
    n_harmonics=60,
    n_harmonics_subset=1,
    delays=[0.0, 0.75])


S11 = npf.sparam(1, 1)
S21 = npf.sparam(2, 1)
S12 = npf.sparam(1, 2)
S21_8th = npf.sparam(2, 1, harmonic=8)
S21_16th = npf.sparam(2, 1, harmonic=16)
S21_32th = npf.sparam(2, 1, harmonic=32)


plt.figure()
plt.plot(freqs/GHz, mag2db(np.abs(S11)), label='$S_{11}$')
plt.plot(freqs/GHz, mag2db(np.abs(S12)), label='$S_{12}$')
plt.plot(freqs/GHz, mag2db(np.abs(S21)), label='$S_{21}$')
plt.plot(freqs/GHz, mag2db(np.abs(S21_8th)), label='$S_{21}$ (8-th)')
plt.plot(freqs/GHz, mag2db(np.abs(S21_16th)), label='$S_{21}$ (16-th)')
plt.plot(freqs/GHz, mag2db(np.abs(S21_32th)), label='$S_{21}$ (32-th)')

plt.xlabel('Frequency (GHz)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.grid()
plt.show()
