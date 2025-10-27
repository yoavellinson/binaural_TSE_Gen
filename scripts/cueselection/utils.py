import numpy as np
import numpy as np
import scipy.signal as signal


def ild_itd_ic(left_signal, right_signal, fs, critical_band_freqs):
    num_bands = len(critical_band_freqs)
    ic = np.zeros(num_bands)
    itd = np.zeros(num_bands)
    ild = np.zeros(num_bands)

    for i, f in enumerate(critical_band_freqs):
        # Define filter cutoffs ensuring they are within valid range (0, 1)
        low_cutoff = max(0.01, (f - 0.2 * f) / (fs / 2))
        high_cutoff = min(0.99, (f + 0.2 * f) / (fs / 2))

        if low_cutoff >= high_cutoff:
            continue  # Skip invalid filter ranges

        # Bandpass filter using Butterworth (approx. Gammatone)
        b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
        left_band = signal.filtfilt(b, a, left_signal)
        right_band = signal.filtfilt(b, a, right_signal)

        # Compute Interaural Coherence (IC)
        cross_corr = np.correlate(left_band, right_band, mode='full')
        ic[i] = np.max(np.abs(cross_corr)) / (np.sqrt(np.sum(left_band ** 2)) * np.sqrt(np.sum(right_band ** 2)))

        # Compute ITD (Interaural Time Difference)
        lag = np.argmax(np.abs(cross_corr)) - (len(left_band) - 1)
        itd[i] = lag / fs  # Convert to seconds

        # Compute ILD (Interaural Level Difference)
        left_power = np.mean(left_band ** 2)
        right_power = np.mean(right_band ** 2)
        ild[i] = 10 * np.log10(left_power / right_power)

    return ild,itd,ic