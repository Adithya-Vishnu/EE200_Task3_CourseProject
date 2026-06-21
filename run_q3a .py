"""
EE200 Q3A - Complete Execution Script

This script executes the complete Q3A analysis pipeline:
1. Builds the song database from audio files
2. Demonstrates all signal processing phases
3. Tests matching with paired vs single peaks
4. Runs robustness experiments
5. Generates all plots for the report

Run this script to generate all Q3A results.
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import warnings
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
warnings.filterwarnings('ignore')

# Import our modules
from fingerprint import (
    load_audio, plot_dft_failure, plot_spectrogram,
    plot_window_comparison, extract_peaks, plot_constellation,
    create_hashes, create_single_peak_hashes, compute_spectrogram
)
from database_builder import SongDatabase
from matcher import SongMatcher


def ensure_output_directory():
    """Create output directory for plots."""
    os.makedirs('q3a_output', exist_ok=True)


def phase_1_dft_analysis():
    """
    PHASE 1: Demonstrate why single FFT fails.
    
    Load a song and compute its DFT. Show that while we can see
    which frequencies exist, we lose all timing information.
    """
    print("\n" + "="*70)
    print("PHASE 1: Why Single DFT Fails")
    print("="*70)
    
    song_path = "EE200 Project Song Database/Bohemian Rhapsody.mp3"
    if not os.path.exists(song_path):
        print(f"Song not found: {song_path}")
        return
    
    # Load audio
    y, sr, duration = load_audio(song_path)
    print(f"\nLoaded: {Path(song_path).stem}")
    print(f"Duration: {duration:.1f} seconds")
    print(f"Sampling rate: {sr} Hz")
    
    # Plot DFT failure
    print("\nComputing DFT...")
    fig = plot_dft_failure(y, sr)
    plt.tight_layout()
    fig.savefig('q3a_output/01_dft_failure.png', dpi=150, bbox_inches='tight')
    print(" Saved: 01_dft_failure.png")
    plt.close(fig)


def phase_2_spectrogram_analysis():
    """
    PHASE 2: Compute and analyze spectrograms.
    
    1. Compute spectrogram of a song
    2. Compare short window vs long window
    3. Demonstrate time-frequency tradeoff
    """
    print("\n" + "="*70)
    print("PHASE 2: Spectrogram & Time-Frequency Tradeoff")
    print("="*70)
    
    song_path = "EE200 Project Song Database/Bohemian Rhapsody.mp3"
    
    # Load audio
    y, sr, duration = load_audio(song_path)
    print(f"\nAnalyzing: {Path(song_path).stem}")
    
    # Plot default spectrogram
    print("\nComputing standard spectrogram (n_fft=2048)...")
    fig, S_db, freqs, times = plot_spectrogram(y, sr, title=
        "Standard Spectrogram (n_fft=2048)\n" +
        "Good balance between time and frequency resolution")
    fig.savefig('q3a_output/02_spectrogram_default.png', dpi=150, bbox_inches='tight')
    print(" Saved: 02_spectrogram_default.png")
    plt.close(fig)
    
    # Plot window comparison
    print("Computing window comparison (short vs long)...")
    fig = plot_window_comparison(y, sr)
    fig.savefig('q3a_output/03_window_comparison.png', dpi=150, bbox_inches='tight')
    print(" Saved: 03_window_comparison.png")
    plt.close(fig)
    
    # Create explanation text
    explanation = """
    KEY OBSERVATION - TIME-FREQUENCY TRADEOFF:
    
    SHORT WINDOW (512 samples):
    • PROS: Excellent time resolution → see WHEN each note occurs
    • CONS: Poor frequency resolution → frequencies appear blurry
    • Use case: Detecting rapid changes, note onsets
    
    LONG WINDOW (4096 samples):
    • PROS: Excellent frequency resolution → see WHICH notes clearly
    • CONS: Poor time resolution → lose timing precision
    • Use case: Identifying steady tones, sustained notes
    
    DEFAULT (2048 samples):
    • Compromise between both
    • Good enough for our fingerprinting purpose
    
    This is a fundamental property of signal processing (Heisenberg
    Uncertainty Principle): you cannot have arbitrarily good resolution
    in both time and frequency simultaneously.
    """
    print(explanation)


def phase_3_peak_extraction():
    """
    PHASE 3: Extract constellation of peaks from spectrogram.
    
    Find local maxima - these are the distinctive points we use
    for fingerprinting.
    """
    print("\n" + "="*70)
    print("PHASE 3: Constellation Peak Extraction")
    print("="*70)
    
    song_path = "EE200 Project Song Database/Bohemian Rhapsody.mp3"
    
    # Load and process
    y, sr, duration = load_audio(song_path)
    S_db, freqs, times = compute_spectrogram(y, sr)
    peaks, peak_indices = extract_peaks(S_db, freqs, times)
    
    print(f"\nDetected {len(peaks)} constellation peaks")
    print(f"Peak distribution (top 5 by amplitude):")
    for i, (freq, time, amp) in enumerate(peaks[:5]):
        print(f"  {i+1}. Freq: {freq:.0f} Hz, Time: {time:.2f}s, Amp: {amp:.1f} dB")
    
    # Plot constellation
    print("\nPlotting constellation...")
    fig = plot_constellation(S_db, freqs, times, peaks, peak_indices,
                            song_name=Path(song_path).stem)
    fig.savefig('q3a_output/04_constellation_map.png', dpi=150, bbox_inches='tight')
    print(" Saved: 04_constellation_map.png")
    plt.close(fig)


def phase_4_fingerprint_creation():
    """
    PHASE 4: Create fingerprint hashes from peaks.
    
    Pair peaks together to create hashes: (freq1, freq2, time_difference)
    """
    print("\n" + "="*70)
    print("PHASE 4: Fingerprint Hash Creation")
    print("="*70)
    
    song_path = "EE200 Project Song Database/Bohemian Rhapsody.mp3"
    
    # Load and process
    y, sr, duration = load_audio(song_path)
    S_db, freqs, times = compute_spectrogram(y, sr)
    peaks, _ = extract_peaks(S_db, freqs, times)
    
    # Create hashes
    hashes, hash_data = create_hashes(peaks)
    
    print(f"\nCreated {len(hashes)} hashes from {len(peaks)} peaks")
    print(f"\nExample hashes (peak pairs):")
    for i, hash_tuple in enumerate(hashes[:10]):
        freq1, freq2, time_diff = hash_tuple
        print(f"  {i+1}. ({freq1:>4d} Hz, {freq2:>4d} Hz, {time_diff:>5.2f}s)")
    
    # Show why pairs are better than single peaks
    single_hashes, _ = create_single_peak_hashes(peaks)
    print(f"\nCOMPARISON:")
    print(f"  Single-peak hashes: {len(single_hashes)} (many duplicates across songs)")
    print(f"  Pair-based hashes:  {len(hashes)} (much more unique/distinctive)")
    
    explanation = """
    WHY PAIRED HASHES ARE SUPERIOR:
    
    SINGLE PEAKS:
    • Many songs share the same frequencies
    • Example: C note (262 Hz) appears in thousands of songs
    • High false-positive rate → many accidental matches
    
    PEAK PAIRS:
    • Combination of (freq1, freq2, time_gap) is much rarer
    • Example: (262 Hz, 330 Hz, 0.5 sec) appears in few songs
    • Much more distinctive → reliable matching
    
    The pair (freq1, freq2, Δt) forms a "fingerprint" that uniquely
    identifies segments of music.
    """
    print(explanation)


def phase_5_database_building():
    """
    PHASE 5: Build complete song database.
    
    Index all songs and create fingerprint database.
    """
    print("\n" + "="*70)
    print("PHASE 5: Database Building")
    print("="*70)
    
    db = SongDatabase()
    song_dir = "EE200 Project Song Database"
    
    if not os.path.exists(song_dir):
        print(f"Directory not found: {song_dir}")
        return None
    
    print(f"\nIndexing all songs from: {song_dir}\n")
    db.batch_index(song_dir)
    db.print_statistics()
    
    # Save database
    db.save('song_database.pkl')
    
    return db


def phase_6_matching_demonstration(db):
    """
    PHASE 6: Demonstrate song matching.
    
    Show paired-hash matching vs single-peak matching.
    """
    print("\n" + "="*70)
    print("PHASE 6: Song Matching Demonstration")
    print("="*70)
    
    if db is None:
        print("Loading database...")
        db = SongDatabase()
        db.load('song_database.pkl')
    
    # Use one of the Beatles songs as a test
    test_song = "EE200 Project Song Database/Hey Jude.mp3"
    
    if not os.path.exists(test_song):
        print(f"Test song not found: {test_song}")
        return
    
    matcher = SongMatcher(db)
    
    print(f"\nTest song: {Path(test_song).stem}\n")
    
    # Test paired-hash matching
    print("METHOD 1: PAIRED-HASH MATCHING")
    print("-" * 70)
    result_paired = matcher.match_query(test_song, use_single_peaks=False)
    
    if result_paired['matched_song']:
        print(f"✅ MATCHED: {result_paired['matched_song']}")
        print(f"   Confidence: {result_paired['confidence']:.0f} hashes")
        print(f"   Top 5 matches:")
        sorted_songs = sorted(result_paired['song_scores'].items(),
                            key=lambda x: x[1], reverse=True)
        for song, score in sorted_songs[:5]:
            marker = "★" if song == result_paired['matched_song'] else " "
            print(f"   {marker} {song}: {score:.0f}")
    else:
        print("❌ No match found")
    
    # # Test single-peak matching
    # print("\n\nMETHOD 2: SINGLE-PEAK MATCHING (for comparison)")
    # print("-" * 70)
    # result_single = matcher.match_query(test_song, use_single_peaks=True)
    
    # if result_single['matched_song']:
    #     print(f"✅ MATCHED: {result_single['matched_song']}")
    #     print(f"   Confidence: {result_single['confidence']:.0f} peaks")
    # else:
    #     print("❌ No match found")
    
    # Plot offset histogram for correct song
    print("\nGenerating offset histogram...")
    fig, offsets = matcher.plot_offset_histogram(test_song, 
                                                result_paired['matched_song'])
    fig.savefig('q3a_output/05_offset_histogram.png', dpi=150, bbox_inches='tight')
    print("okay Saved: 05_offset_histogram.png")
    plt.close(fig)
    
    explanation = """
    INTERPRETATION OF OFFSET HISTOGRAM:
    
    The histogram shows time offsets where query hashes match database hashes.
    
    CORRECT MATCH:
    • Sharp spike: thousands of hashes align at same offset
    • High peak = high confidence
    • Example: all 500+ matching hashes have offset ~35 seconds
    
    WRONG MATCH:
    • Scattered distribution: random offsets
    • Low peak = low confidence
    • Different hashes match at many different times (coincidence)
    
    This histogram IS the decision-maker for song identification.
    """
    print(explanation)


def main():
    """Execute complete Q3A analysis."""
    
    print("\n" + "="*70)
    print("EE200 Q3A - SONG RECOGNITION SYSTEM")
    print("Complete Analysis Pipeline")
    print("="*70)
    
    # Setup
    ensure_output_directory()
    
    # Execute phases
    phase_1_dft_analysis()
    phase_2_spectrogram_analysis()
    phase_3_peak_extraction()
    phase_4_fingerprint_creation()
    db = phase_5_database_building()
    phase_6_matching_demonstration(db)
    
    print("\n" + "="*70)
    print("Q3A ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  01_dft_failure.png")
    print("  02_spectrogram_default.png")
    print("  03_window_comparison.png")
    print("  04_constellation_map.png")
    print("  05_offset_histogram.png")
    print("\nAll plots saved to: q3a_output/")
    print("\nNext steps:")
    print("  1. Review the plots")
    print("  2. Run robustness tests (q3a_experiments.py)")
    print("  3. Create report with these findings")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     db = phase_5_database_building()