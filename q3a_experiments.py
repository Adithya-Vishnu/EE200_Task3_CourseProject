"""
EE200 Q3A - Robustness Experiments

This module tests how robust the song identification system is against:
1. Additive noise (SNR degradation)
2. Pitch shifting (frequency changes)
3. Time stretching (tempo changes)

These experiments demonstrate why certain aspects of the fingerprinting
system work or fail.
"""

import numpy as np
import librosa
import librosa.effects
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from fingerprint import (
    load_audio, compute_spectrogram, extract_peaks,
    create_hashes, N_FFT, HOP_LENGTH,
    PEAK_NEIGHBORHOOD, PEAK_AMP_MIN
)
from matcher import SongMatcher


# ============================================================================
# NOISE ROBUSTNESS TEST
# ============================================================================

def add_gaussian_noise(y, snr_db):
    """
    Add Gaussian noise to audio at a specified SNR.
    
    Parameters
    ----------
    y : np.ndarray
        Audio signal
    snr_db : float
        Signal-to-noise ratio in dB
        Higher SNR = less noise
        Example: SNR=30dB -> high quality
                 SNR=10dB -> heavy noise
    
    Returns
    -------
    y_noisy : np.ndarray
        Signal with added noise
    """
    signal_power = np.mean(y ** 2)
    snr_ratio = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_ratio
    noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    return y + noise


def test_noise_robustness(db, song_file, snr_values=[40, 30, 20, 10, 5, 0]):
    """
    Test song identification under increasing noise levels.
    
    Parameters
    ----------
    db : SongDatabase
        The database
    song_file : str
        Path to test song
    snr_values : list
        SNR levels (dB) to test
    
    Returns
    -------
    results : dict
        Results for each SNR level
    """
    
    # Load original audio
    y, sr, duration = load_audio(song_file)
    
    # Get the correct song name
    correct_song = Path(song_file).stem
    
    results = {}
    matcher = SongMatcher(db)
    
    print(f"\nNOISE ROBUSTNESS TEST: {correct_song}")
    print("=" * 70)
    print(f"{'SNR (dB)':<12} {'Matched Song':<30} {'Confidence':<15} {'Status':<15}")
    print("-" * 70)
    
    for snr in snr_values:
        # Add noise
        y_noisy = add_gaussian_noise(y, snr)
        
        # Create temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import soundfile as sf
            try:
                sf.write(tmp.name, y_noisy, sr)
                
                # Test matching
                result = matcher.match_query(tmp.name, use_single_peaks=False)
                
                matched = result['matched_song'] if result['matched_song'] else "None"
                confidence = result['confidence']
                
                is_correct = (matched == correct_song)
                status = " PASS" if is_correct else " FAIL"
                
                results[snr] = {
                    'matched': matched,
                    'confidence': confidence,
                    'correct': is_correct
                }
                
                print(f"{snr:<12} {matched:<30} {confidence:<15.0f} {status:<15}")
                
                # Clean up
                import os
                # os.unlink(tmp.name)
                
            except ImportError:
                print("  (soundfile not installed - skipping noise test)")
                break
    
    print("=" * 70)
    
    return results


def plot_noise_results(noise_results, snr_values):
    """
    Plot noise robustness results.
    
    Parameters
    ----------
    noise_results : dict
        Results from test_noise_robustness
    snr_values : list
        SNR values tested
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    
    successes = [1 if noise_results[snr]['correct'] else 0 for snr in snr_values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(snr_values, successes, 'o-', markersize=8, linewidth=2,
           color='steelblue', label='Recognition Success')
    ax.fill_between(snr_values, successes, alpha=0.3, color='steelblue')
    
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Recognition Success (1=Yes, 0=No)', fontsize=12)
    ax.set_title('Noise Robustness: How Much Noise Can We Tolerate?',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.1, 1.1])
    ax.set_xticks(snr_values)
    
    # Add threshold annotation
    last_success_idx = -1
    for i in range(len(successes)-1, -1, -1):
        if successes[i] == 1:
            last_success_idx = i
            break
    
    if last_success_idx >= 0:
        threshold_snr = snr_values[last_success_idx]
        ax.axvline(threshold_snr, color='red', linestyle='--', alpha=0.5)
        ax.text(threshold_snr, 0.95, f'Recognition Threshold\n~{threshold_snr} dB',
               ha='center', fontsize=10, bbox=dict(boxstyle='round', 
               facecolor='yellow', alpha=0.3))
    
    return fig


# ============================================================================
# PITCH SHIFT TEST
# ============================================================================

def test_pitch_shift(db, song_file, pitch_shifts=[-2, -1, 0, 1, 2]):
    """
    Test song identification with pitch-shifted versions.
    
    This demonstrates a KEY LIMITATION: the system uses ABSOLUTE frequencies.
    When you shift pitch, all frequencies move, breaking the hashes.
    
    Parameters
    ----------
    db : SongDatabase
        The database
    song_file : str
        Path to test song
    pitch_shifts : list
        Semitone shifts to test (0 = original)
    
    Returns
    -------
    results : dict
        Results for each pitch shift
    """
    
    # Load original audio
    y, sr, duration = load_audio(song_file)
    correct_song = Path(song_file).stem
    
    results = {}
    matcher = SongMatcher(db)
    
    print(f"\nPITCH SHIFT TEST: {correct_song}")
    print("=" * 70)
    print(f"{'Shift (semitones)':<20} {'Matched Song':<30} {'Confidence':<15} {'Status':<15}")
    print("-" * 70)
    
    for semitones in pitch_shifts:
        try:
            # Pitch shift
            y_shifted = librosa.effects.pitch_shift(
                    y,
                    sr=sr,
                    n_steps=semitones
                )
            
            # Create temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                import soundfile as sf
                try:
                    sf.write(tmp.name, y_shifted, sr)
                    
                    # Test matching
                    result = matcher.match_query(tmp.name, use_single_peaks=False)
                    
                    matched = result['matched_song'] if result['matched_song'] else "None"
                    confidence = result['confidence']
                    
                    is_correct = (matched == correct_song)
                    status = " PASS" if is_correct else " FAIL"
                    
                    results[semitones] = {
                        'matched': matched,
                        'confidence': confidence,
                        'correct': is_correct
                    }
                    
                    # Display frequency shift info
                    freq_shift = semitones * 5.95  # ~6% per semitone
                    print(f"{semitones:<20} {matched:<30} {confidence:<15.0f} {status:<15}")
                    
                    # Clean up
                    import os
                    # os.unlink(tmp.name)
                    
                except ImportError:
                    print("  (soundfile not installed - skipping pitch shift test)")
                    break
        
        except Exception as e:
            print(f"  Shift {semitones}: Error - {str(e)}")
    
    print("=" * 70)
    


    return results


def plot_pitch_results(pitch_results, pitch_shifts):
    """
    Plot pitch shift robustness results.
    
    Parameters
    ----------
    pitch_results : dict
        Results from test_pitch_shift
    pitch_shifts : list
        Pitch shifts tested
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    
    successes = [1 if pitch_results[s]['correct'] else 0 for s in pitch_shifts]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['red' if s != 0 else 'green' for s in pitch_shifts]
    ax.bar(range(len(pitch_shifts)), successes, color=colors, alpha=0.7,
          edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Pitch Shift (semitones)', fontsize=12)
    ax.set_ylabel('Recognition Success', fontsize=12)
    ax.set_title('Pitch Shift Robustness: System FAILS with ANY pitch change',
                fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(pitch_shifts)))
    ax.set_xticklabels(pitch_shifts)
    ax.set_ylim([0, 1.2])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add annotation
    ax.text(0.5, 0.95, 'Why it fails: All frequencies change → hashes break',
           ha='center', va='top', transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
           fontsize=10)
    
    return fig


# ============================================================================
# TIME STRETCH TEST
# ============================================================================

def test_time_stretch(db, song_file, stretch_factors=[0.95, 0.98, 1.0, 1.02, 1.05]):
    """
    Test song identification with time-stretched versions.
    
    Time stretching changes the time differences between peaks,
    breaking the time-difference component of the hashes.
    
    Parameters
    ----------
    db : SongDatabase
        The database
    song_file : str
        Path to test song
    stretch_factors : list
        Stretch factors to test (1.0 = original, 1.05 = 5% faster)
    
    Returns
    -------
    results : dict
        Results for each stretch factor
    """
    
    # Load original audio
    y, sr, duration = load_audio(song_file)
    correct_song = Path(song_file).stem
    
    results = {}
    matcher = SongMatcher(db)
    
    print(f"\nTIME STRETCH TEST: {correct_song}")
    print("=" * 70)
    print(f"{'Stretch Factor':<20} {'Matched Song':<30} {'Confidence':<15} {'Status':<15}")
    print("-" * 70)
    
    for factor in stretch_factors:
        try:
            # Time stretch
            y_stretched = librosa.effects.time_stretch(
        y,
        rate=factor
    )
            
            # Create temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                import soundfile as sf
                try:
                    sf.write(tmp.name, y_stretched, sr)
                    
                    # Test matching
                    result = matcher.match_query(tmp.name, use_single_peaks=False)
                    
                    matched = result['matched_song'] if result['matched_song'] else "None"
                    confidence = result['confidence']
                    
                    is_correct = (matched == correct_song)
                    status = " PASS" if is_correct else " FAIL"
                    
                    results[factor] = {
                        'matched': matched,
                        'confidence': confidence,
                        'correct': is_correct
                    }
                    
                    percent_change = (factor - 1) * 100
                    print(f"{factor:<20} {matched:<30} {confidence:<15.0f} {status:<15}")
                    
                    # Clean up
                    import os
                    # os.unlink(tmp.name)
                    
                except ImportError:
                    print("  (soundfile not installed - skipping time stretch test)")
                    break
        
        except Exception as e:
            print(f"  Factor {factor}: Error - {str(e)}")
    
    print("=" * 70)
    
    return results


def plot_time_stretch_results(stretch_results, stretch_factors):
    """
    Plot time stretch robustness results.
    
    Parameters
    ----------
    stretch_results : dict
        Results from test_time_stretch
    stretch_factors : list
        Stretch factors tested
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    
    successes = [1 if stretch_results[f]['correct'] else 0 for f in stretch_factors]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    percent_changes = [(f - 1) * 100 for f in stretch_factors]
    colors = ['red' if p != 0 else 'green' for p in percent_changes]
    
    ax.bar(range(len(stretch_factors)), successes, color=colors, alpha=0.7,
          edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Tempo Change (%)', fontsize=12)
    ax.set_ylabel('Recognition Success', fontsize=12)
    ax.set_title('Time Stretch Robustness: System sensitive to tempo changes',
                fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(stretch_factors)))
    ax.set_xticklabels([f'{p:+.0f}%' for p in percent_changes], rotation=0)
    ax.set_ylim([0, 1.2])
    ax.grid(True, alpha=0.3, axis='y')
    
    return fig


# ============================================================================
# COMPARISON SUMMARY
# ============================================================================

def create_robustness_summary(noise_results, pitch_results, time_stretch_results):
    """
    Create a summary comparison of all robustness tests.
    
    Parameters
    ----------
    noise_results, pitch_results, time_stretch_results : dict
        Results from the respective tests
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Summary figure
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    # Noise
    snr_values = sorted(noise_results.keys())
    noise_success = [1 if noise_results[s]['correct'] else 0 for s in snr_values]
    axes[0].plot(snr_values, noise_success, 'o-', markersize=8, linewidth=2, color='steelblue')
    axes[0].fill_between(snr_values, noise_success, alpha=0.3, color='steelblue')
    axes[0].set_xlabel('SNR (dB)', fontsize=11)
    axes[0].set_ylabel('Success', fontsize=11)
    axes[0].set_title('Noise Robustness', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([-0.1, 1.1])
    
    # Pitch
    pitch_values = sorted(pitch_results.keys())
    pitch_success = [1 if pitch_results[p]['correct'] else 0 for p in pitch_values]
    axes[1].bar(range(len(pitch_values)), pitch_success, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Pitch Shift (semitones)', fontsize=11)
    axes[1].set_ylabel('Success', fontsize=11)
    axes[1].set_title('Pitch Shift Robustness', fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(len(pitch_values)))
    axes[1].set_xticklabels(pitch_values)
    axes[1].set_ylim([0, 1.2])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Time stretch
    stretch_values = sorted(time_stretch_results.keys())
    stretch_success = [1 if time_stretch_results[s]['correct'] else 0 for s in stretch_values]
    axes[2].bar(range(len(stretch_values)), stretch_success, color='lightgreen', 
               alpha=0.7, edgecolor='black')
    percent_changes = [(s - 1) * 100 for s in stretch_values]
    axes[2].set_xlabel('Tempo Change (%)', fontsize=11)
    axes[2].set_ylabel('Success', fontsize=11)
    axes[2].set_title('Time Stretch Robustness', fontsize=12, fontweight='bold')
    axes[2].set_xticks(range(len(stretch_values)))
    axes[2].set_xticklabels([f'{p:+.0f}%' for p in percent_changes], rotation=0)
    axes[2].set_ylim([0, 1.2])
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Robustness Testing Summary', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig
