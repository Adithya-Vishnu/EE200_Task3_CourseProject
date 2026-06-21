"""
EE200 Q3A - Song Matcher

This module performs the actual song identification:
1. Fingerprint the query clip
2. Look up hashes in database
3. Compute time offsets
4. Identify song using offset histogram

The key insight: when a correct match is found, thousands of hashes
will align at the SAME time offset, creating a massive spike in the
offset histogram. Wrong songs produce scattered random offsets.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from fingerprint import (
    load_audio, compute_spectrogram, extract_peaks,
    create_hashes, create_single_peak_hashes,
    N_FFT, HOP_LENGTH, PEAK_NEIGHBORHOOD, PEAK_AMP_MIN
)


class SongMatcher:
    """
    Identifies songs by matching fingerprints against a database.
    """
    
    def __init__(self, database):
        """
        Initialize matcher with a database.
        
        Parameters
        ----------
        database : SongDatabase
            The fingerprint database
        """
        self.database = database.database
        self.single_peak_db = database.single_peak_db
        self.song_list = database.song_list
    
    def match_query(self, query_path, use_single_peaks=False, 
                    confidence_threshold=40):
        """
        Identify a song from a query clip.
        
        Parameters
        ----------
        query_path : str
            Path to query audio file
        use_single_peaks : bool
            If True, use single-peak matching (for comparison)
            If False, use pair-based matching
        confidence_threshold : int
            Minimum number of matching hashes required
        
        Returns
        -------
        result : dict
            {
                'matched_song': song_name or None,
                'confidence': confidence_score,
                'offsets': offset_array,
                'matched_song_offsets': offsets_for_matched_song,
                'song_scores': {song: score for all songs},
                'query_hashes': hashes from query,
                'peaks': detected peaks,
                'S_db': spectrogram
            }
        """
        
        # Load and fingerprint query
        y, sr, duration = load_audio(query_path)
        S_db, freqs, times = compute_spectrogram(y, sr)
        peaks, _ = extract_peaks(S_db, freqs, times,
                                 neighborhood=PEAK_NEIGHBORHOOD,
                                 amp_min=PEAK_AMP_MIN)
        
        if len(peaks) < 5:
            return {
                'matched_song': None,
                'confidence': 0,
                'reason': 'Not enough peaks detected in query',
                'offsets': np.array([]),
                'query_hashes': [],
                'peaks': peaks,
                'S_db': S_db
            }
        
        # Create hashes from query
        if use_single_peaks:
            query_hashes, _ = create_single_peak_hashes(peaks)
            db_to_search = self.single_peak_db
        else:
            query_hashes, _ = create_hashes(peaks)
            db_to_search = self.database
        
        # Look up hashes in database and collect offsets
        offsets_by_song = {song: [] for song in self.song_list}
        total_matches = 0
        
        for query_hash in query_hashes:
            if query_hash in db_to_search:
                database_entries = db_to_search[query_hash]
                total_matches += len(database_entries)
                
                for db_song_name, db_time in database_entries:
                    # Get the time of this hash in the query
                    # For single peaks, use the peak time directly
                    if use_single_peaks:
                        # Find the corresponding peak
                        query_peak_idx = query_hashes.index(query_hash)
                        query_time = peaks[query_peak_idx][1]
                    else:
                        query_time = 0  # Will be computed from hash
                    
                    # Compute offset: where would this match occur in the original song?
                    # offset = database_time - query_time (approximately)
                    offset = db_time
                    offsets_by_song[db_song_name].append(offset)
        
        # Create offset histogram for each song
        song_scores = {}
        matched_song_offsets = []
        
        for song_name in self.song_list:
            offsets = np.array(offsets_by_song[song_name])
            
            if len(offsets) == 0:
                song_scores[song_name] = 0
            else:
                # Histogram of offsets: sharp spike = good match
                hist, bins = np.histogram(offsets, bins=100)
                song_scores[song_name] = np.max(hist)  # Height of tallest peak
                
                if song_name == self.song_list[0]:  # Save first song's offsets for plotting
                    matched_song_offsets = offsets
        
        # Find best match
        best_song = max(song_scores, key=song_scores.get)
        best_score = song_scores[best_song]

        # print(f"Best song: {best_song}")
        # print(f"Best score: {best_score}")

        if best_score < confidence_threshold:
            matched_song = None
            confidence = 0
        else:
            matched_song = best_song
            confidence = best_score
        
        return {
    'matched_song': matched_song,
    'confidence': confidence,
    'song_scores': song_scores,
    'offsets': matched_song_offsets,
    'query_hashes': query_hashes,
    'peaks': peaks,
    'S_db': S_db,
    'freqs': freqs,
    'times': times,
    'use_single_peaks': use_single_peaks
}
    
    def plot_offset_histogram(self, query_path, matched_song=None):
        """
        Plot offset histogram showing match confidence.
        
        A correct match: one huge peak (all hashes align)
        A wrong match: scattered, random distribution
        
        Parameters
        ----------
        query_path : str
            Path to query file
        matched_song : str, optional
            Song to compute histogram for
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        offsets : np.ndarray
            The offset values (for the matched song)
        """
        
        # Fingerprint query
        y, sr, duration = load_audio(query_path)
        S_db, freqs, times = compute_spectrogram(y, sr)
        peaks, _ = extract_peaks(S_db, freqs, times,
                                 neighborhood=PEAK_NEIGHBORHOOD,
                                 amp_min=PEAK_AMP_MIN)
        
        query_hashes, query_hash_data = create_hashes(peaks)
        
        # Collect offsets for the specified song
        offsets = []
        for query_hash in query_hashes:
            if query_hash in self.database:
                for db_song, db_time in self.database[query_hash]:
                    if matched_song is None or db_song == matched_song:
                        offsets.append(db_time)
        
        # Plot histogram
        fig, ax = plt.subplots(figsize=(12, 5))
        
        if len(offsets) > 0:
            ax.hist(offsets, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_ylabel('Number of Matching Hashes', fontsize=11)
            ax.set_xlabel('Time Offset (seconds)', fontsize=11)
            
            if matched_song:
                max_count = np.max(np.histogram(offsets, bins=50)[0])
                ax.set_title(f'Offset Histogram: {matched_song}\n' +
                            f'Peak height = {max_count} matching hashes = Confidence score',
                            fontsize=12, fontweight='bold')
            else:
                ax.set_title('Offset Histogram',
                           fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No matches found', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Offset Histogram: No Matches',
                        fontsize=12, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, None])
        
        return fig, np.array(offsets)


def demonstrate_matching(db, query_song_name):
    """
    Demonstrate both paired-hash and single-peak matching on the same query.
    
    Parameters
    ----------
    db : SongDatabase
        The database
    query_song_name : str
        Name of song to use as query (must be in database)
    
    Returns
    -------
    results : dict
        Results from both matching methods
    """
    
    # Find the query song file
    song_dir = Path("EE200 Project Song Database")
    query_file = None
    
    for audio_file in song_dir.glob("*"):
        if audio_file.stem == query_song_name:
            query_file = str(audio_file)
            break
    
    if query_file is None:
        print(f"Song not found: {query_song_name}")
        return None
    
    matcher = SongMatcher(db)
    
    print(f"\n{'='*70}")
    print(f"MATCHING DEMONSTRATION: {query_song_name}")
    print(f"{'='*70}\n")
    
    # Method 1: Paired hashes
    print("METHOD 1: PAIRED-HASH MATCHING")
    print("-" * 70)
    result_paired = matcher.match_query(query_file, use_single_peaks=False)
    
    if result_paired['matched_song']:
        print(f"✅ MATCHED: {result_paired['matched_song']}")
        print(f"   Confidence: {result_paired['confidence']:.0f} hashes")
        print(f"   Matched songs (top 5):")
        sorted_scores = sorted(result_paired['song_scores'].items(),
                             key=lambda x: x[1], reverse=True)
        for song, score in sorted_scores[:5]:
            print(f"      - {song}: {score:.0f}")
    else:
        print(f"❌ NO MATCH FOUND")
    
    # Method 2: Single peaks
    print("\n\nMETHOD 2: SINGLE-PEAK MATCHING (for comparison)")
    print("-" * 70)
    result_single = matcher.match_query(query_file, use_single_peaks=True)
    
    if result_single['matched_song']:
        print(f"✅ MATCHED: {result_single['matched_song']}")
        print(f"   Confidence: {result_single['confidence']:.0f} peaks")
    else:
        print(f"❌ NO MATCH FOUND")
    
    print(f"\n{'='*70}\n")
    
    return {
        'paired': result_paired,
        'single': result_single
    }
