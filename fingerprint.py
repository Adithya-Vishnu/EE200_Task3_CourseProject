"""
EE200 Q3A - Song Recognition System: Core Fingerprinting Module

This module implements the complete fingerprinting pipeline:
1. Audio loading and preprocessing
2. Spectrogram computation (time-frequency analysis)
3. Peak detection (constellation extraction)
4. Combinatorial hashing (peak pair hashing)

Author: Signal Processing Implementation
Date: 2026
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import maximum_filter
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Spectrogram parameters
SAMPLE_RATE = 22050          # Standard audio sampling rate
N_FFT = 2048                 # FFT window size (default: good time-frequency balance)
HOP_LENGTH = 512             # Number of samples between successive frames
N_FFT_SHORT = 512            # Short window for time resolution experiment
N_FFT_LONG = 4096            # Long window for frequency resolution experiment

# Peak detection parameters
PEAK_NEIGHBORHOOD = 25       # Neighborhood size for local maxima detection
PEAK_AMP_MIN = -60          # Minimum amplitude threshold (in dB)

# Hash pairing parameters
TARGET_ZONE_MIN = 0.5        # Minimum time gap between paired peaks (seconds)
TARGET_ZONE_MAX = 15         # Maximum time gap between paired peaks (seconds)
HASH_TIME_QUANTIZE = 0.5     # Quantize time difference to 0.5 second bins


# ============================================================================
# PHASE 1: AUDIO LOADING AND DFT ANALYSIS
# ============================================================================

def load_audio(song_path):
    """
    Load audio file using librosa.
    
    Parameters
    ----------
    song_path : str
        Path to audio file (mp3, wav, etc.)
    
    Returns
    -------
    y : np.ndarray
        Audio time series
    sr : int
        Sampling rate (typically 22050 Hz)
    duration : float
        Duration in seconds
    """
    y, sr = librosa.load(song_path, sr=SAMPLE_RATE)
    duration = librosa.get_duration(y=y, sr=sr)
    return y, sr, duration


def compute_dft(y):
    """
    Compute the Discrete Fourier Transform of the entire song.
    
    This shows WHY a single FFT fails: you can see which frequencies exist,
    but not WHEN they occur.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    
    Returns
    -------
    fft_mag : np.ndarray
        Magnitude spectrum
    freqs : np.ndarray
        Frequency bins (Hz)
    """
    fft = np.fft.rfft(y)
    fft_mag = np.abs(fft)
    freqs = np.fft.rfftfreq(len(y), 1/SAMPLE_RATE)
    return fft_mag, freqs


def plot_dft_failure(y, sr):
    """
    Demonstrate why a single DFT fails: loss of temporal information.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int
        Sampling rate
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the plot
    """
    fft_mag, freqs = compute_dft(y)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot only up to 5000 Hz for clarity
    ax.semilogy(freqs[:len(freqs)//4], fft_mag[:len(fft_mag)//4], linewidth=0.8)
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Magnitude (log scale)', fontsize=12)
    ax.set_title('Why Single FFT Fails: Temporal Information Lost\n' + 
                 'We see WHICH frequencies exist, but not WHEN they occur',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 5000])
    
    return fig


# ============================================================================
# PHASE 2: SPECTROGRAM COMPUTATION (Short-Time Fourier Transform)
# ============================================================================

def compute_spectrogram(y, sr, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Compute spectrogram (time-frequency representation) using STFT.
    
    This shows frequencies as they CHANGE over time - the key insight.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int
        Sampling rate
    n_fft : int
        FFT window size
    hop_length : int
        Samples between successive frames
    
    Returns
    -------
    S_db : np.ndarray
        Spectrogram in dB (power per frequency per time)
    freqs : np.ndarray
        Frequency bins (Hz)
    times : np.ndarray
        Time bins (seconds)
    """
    # Compute STFT (complex-valued)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    
    # Convert to power and then to dB scale for visualization
    S = np.abs(D) ** 2
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Compute frequency and time axes
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), 
                                   sr=sr, hop_length=hop_length)
    
    return S_db, freqs, times


def plot_spectrogram(y, sr, n_fft=N_FFT, hop_length=HOP_LENGTH, title="Spectrogram"):
    """
    Plot a spectrogram with professional formatting.
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int
        Sampling rate
    n_fft : int
        FFT window size
    hop_length : int
        Samples between successive frames
    title : str
        Plot title
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    S_db : np.ndarray
        The computed spectrogram
    freqs : np.ndarray
        Frequency bins
    times : np.ndarray
        Time bins
    """
    S_db, freqs, times = compute_spectrogram(y, sr, n_fft, hop_length)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, 
                                    x_axis='time', y_axis='log', ax=ax,
                                    cmap='magma')
    ax.set_ylim([50, 10000])  # Focus on audible range
    ax.set_ylabel('Frequency (Hz)', fontsize=11)
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.set_label('Power (dB)', fontsize=10)
    
    return fig, S_db, freqs, times


def plot_window_comparison(y, sr):
    """
    Demonstrate the time-frequency tradeoff with short vs long windows.
    
    SHORT WINDOW (512 samples):
    - Good time resolution (know WHEN notes occur)
    - Poor frequency resolution (frequencies are blurry)
    
    LONG WINDOW (4096 samples):
    - Good frequency resolution (know WHICH frequencies occur)
    - Poor time resolution (timing information is blurred)
    
    Parameters
    ----------
    y : np.ndarray
        Audio time series
    sr : int
        Sampling rate
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with side-by-side comparison
    """
    S_short, freqs_short, times_short = compute_spectrogram(
        y, sr, n_fft=N_FFT_SHORT, hop_length=N_FFT_SHORT//4
    )
    S_long, freqs_long, times_long = compute_spectrogram(
        y, sr, n_fft=N_FFT_LONG, hop_length=N_FFT_LONG//4
    )
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Short window
    img1 = librosa.display.specshow(S_short, sr=sr, hop_length=N_FFT_SHORT//4,
                                     x_axis='time', y_axis='log', ax=axes[0],
                                     cmap='magma')
    axes[0].set_ylim([50, 10000])
    axes[0].set_title(
        'SHORT Window (512 samples): Better TIME Resolution, Worse FREQUENCY Resolution\n' +
        'You see WHEN notes start/end precisely, but frequency content is blurred',
        fontsize=11, fontweight='bold', pad=10
    )
    axes[0].set_ylabel('Frequency (Hz)', fontsize=10)
    fig.colorbar(img1, ax=axes[0], format='%+2.0f dB', label='Power (dB)')
    
    # Long window
    img2 = librosa.display.specshow(S_long, sr=sr, hop_length=N_FFT_LONG//4,
                                     x_axis='time', y_axis='log', ax=axes[1],
                                     cmap='magma')
    axes[1].set_ylim([50, 10000])
    axes[1].set_title(
        'LONG Window (4096 samples): Better FREQUENCY Resolution, Worse TIME Resolution\n' +
        'You see WHICH notes exist clearly, but timing details are lost',
        fontsize=11, fontweight='bold', pad=10
    )
    axes[1].set_ylabel('Frequency (Hz)', fontsize=10)
    axes[1].set_xlabel('Time (seconds)', fontsize=10)
    fig.colorbar(img2, ax=axes[1], format='%+2.0f dB', label='Power (dB)')
    
    plt.tight_layout()
    return fig


# ============================================================================
# PHASE 3: PEAK EXTRACTION (Constellation Map)
# ============================================================================

def extract_peaks(S_db, freqs, times, neighborhood=PEAK_NEIGHBORHOOD, 
                  amp_min=PEAK_AMP_MIN):
    """
    Extract local maxima from spectrogram - these form the constellation.
    
    A peak is kept if it's the brightest point in its local neighborhood.
    This reduces millions of spectrogram pixels to thousands of key points.
    
    Parameters
    ----------
    S_db : np.ndarray
        Spectrogram (frequency × time)
    freqs : np.ndarray
        Frequency bins
    times : np.ndarray
        Time bins
    neighborhood : int
        Size of neighborhood for local maxima detection
    amp_min : float
        Minimum amplitude threshold (dB)
    
    Returns
    -------
    peaks : list of tuples
        Each tuple: (frequency, time, amplitude)
    peak_indices : tuple
        (frequency_indices, time_indices) for visualization
    """
    # Apply local maxima filter
    local_max = maximum_filter(S_db, size=neighborhood) == S_db
    
    # Apply amplitude threshold
    local_max &= (S_db > amp_min)
    
    # Find coordinates of peaks
    freq_indices, time_indices = np.where(local_max)
    
    # Convert indices to actual frequency/time values
    peak_freqs = freqs[freq_indices]
    peak_times = times[time_indices]
    peak_amps = S_db[freq_indices, time_indices]
    
    # Sort by amplitude (descending) - strongest peaks first
    sorted_idx = np.argsort(-peak_amps)
    
    peaks = list(zip(
    peak_freqs[sorted_idx],
    peak_times[sorted_idx],
    peak_amps[sorted_idx]
))

    MAX_PEAKS = 15000
    peaks = peaks[:MAX_PEAKS]
    
    return peaks, (freq_indices[sorted_idx], time_indices[sorted_idx])


def plot_constellation(S_db, freqs, times, peaks, peak_indices, 
                       song_name="Unknown"):
    """
    Plot spectrogram with constellation peaks overlaid.
    
    Parameters
    ----------
    S_db : np.ndarray
        Spectrogram
    freqs : np.ndarray
        Frequency bins
    times : np.ndarray
        Time bins
    peaks : list of tuples
        Extracted peaks
    peak_indices : tuple
        (freq_indices, time_indices)
    song_name : str
        Name of song for title
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot spectrogram
    img = librosa.display.specshow(S_db, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                    x_axis='time', y_axis='log', ax=ax,
                                    cmap='magma')
    ax.set_ylim([50, 10000])
    
    # Overlay peaks as scatter plot
    if len(peaks) > 0:
        peak_freqs = np.array([p[0] for p in peaks])
        peak_times = np.array([p[1] for p in peaks])
        ax.scatter(peak_times, peak_freqs, c='cyan', s=30, alpha=0.7,
                  edgecolors='white', linewidth=0.5, label=f'Peaks ({len(peaks)})')
    
    ax.set_ylabel('Frequency (Hz)', fontsize=11)
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_title(f'Constellation Map: {song_name}\n' + 
                 f'Detected {len(peaks)} constellation peaks',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    
    fig.colorbar(img, ax=ax, format='%+2.0f dB', label='Power (dB)')
    
    return fig


# ============================================================================
# PHASE 4: COMBINATORIAL HASHING (Peak Pair Hashing)
# ============================================================================

def create_hashes(peaks, quantize_time=HASH_TIME_QUANTIZE):
    """
    Create hashes from peak pairs.
    
    The key insight: a single frequency is NOT unique (many songs share notes).
    A PAIR of frequencies with specific timing is much rarer.
    
    For each peak (anchor), we look forward to nearby peaks (targets)
    and create a hash: (freq_anchor, freq_target, time_difference)
    
    Parameters
    ----------
    peaks : list of tuples
        (frequency, time, amplitude) for each peak
    quantize_time : float
        Quantize time differences to bins of this size
    
    Returns
    -------
    hashes : list of tuples
        Each hash: (freq1, freq2, time_diff_quantized)
    hash_data : list of tuples
        Each entry: (hash, song_time) for database lookup
    """
    hashes = []
    hash_data = []
    
    n_peaks = len(peaks)
    
    # For each peak as anchor
    for i in range(n_peaks):
        anchor_freq, anchor_time, _ = peaks[i]
        
        # Look forward to nearby peaks (targets)
        for j in range(i+1, min(i+15, n_peaks)):  # Limit pairing distance
            target_freq, target_time, _ = peaks[j]
            
            time_diff = target_time - anchor_time
            
            # Only pair peaks within target zone
            if TARGET_ZONE_MIN <= time_diff <= TARGET_ZONE_MAX:
                # Quantize time difference
                time_diff_quantized = round(time_diff / quantize_time) * quantize_time
                
                # Create hash (ensure consistent ordering)
                if anchor_freq <= target_freq:
                    hash_tuple = (int(anchor_freq), int(target_freq), 
                                 round(time_diff_quantized, 2))
                else:
                    hash_tuple = (int(target_freq), int(anchor_freq),
                                 round(time_diff_quantized, 2))
                
                hashes.append(hash_tuple)
                hash_data.append((hash_tuple, anchor_time))
    
    return hashes, hash_data


def create_single_peak_hashes(peaks):
    """
    Create hashes using ONLY single peaks (no pairing).
    
    This is used for comparison - to show why single peaks are insufficient.
    
    Parameters
    ----------
    peaks : list of tuples
        (frequency, time, amplitude)
    
    Returns
    -------
    single_hashes : list of int
        Just the quantized frequencies
    hash_data : list of tuples
        (hash, song_time) entries
    """
    single_hashes = []
    hash_data = []
    
    for peak_freq, peak_time, _ in peaks:
        freq_hash = int(peak_freq / 100) * 100  # Quantize to 100 Hz bins
        single_hashes.append(freq_hash)
        hash_data.append((freq_hash, peak_time))
    
    return single_hashes, hash_data
