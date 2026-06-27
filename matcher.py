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
from collections import Counter
from pathlib import Path
from time import perf_counter
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
                    confidence_threshold=40, max_duration=30):
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
        
        total_started = perf_counter()

        # Load and fingerprint query (queries are capped for predictable runtime).
        load_started = perf_counter()
        y, sr, duration = load_audio(query_path, max_duration=max_duration)
        load_ms = (perf_counter() - load_started) * 1000

        spectrogram_started = perf_counter()
        S_db, freqs, times = compute_spectrogram(y, sr)
        spectrogram_ms = (perf_counter() - spectrogram_started) * 1000

        constellation_started = perf_counter()
        peaks, _ = extract_peaks(S_db, freqs, times,
                                 neighborhood=PEAK_NEIGHBORHOOD,
                                 amp_min=PEAK_AMP_MIN)
        constellation_ms = (perf_counter() - constellation_started) * 1000
        
        if len(peaks) < 5:
            total_ms = (perf_counter() - total_started) * 1000
            return {
                'matched_song': None,
                'confidence': 0,
                'reason': 'Not enough peaks detected in query',
                'offsets': np.array([]),
                'song_scores': {song: 0 for song in self.song_list},
                'query_hashes': [],
                'peaks': peaks,
                'S_db': S_db,
                'freqs': freqs,
                'times': times,
                'duration': duration,
                'total_hash_matches': 0,
                'use_single_peaks': use_single_peaks,
                'timings': {
                    'audio_load_ms': load_ms,
                    'spectrogram_ms': spectrogram_ms,
                    'constellation_ms': constellation_ms,
                    'hash_generation_ms': 0.0,
                    'database_lookup_ms': 0.0,
                    'scoring_ms': 0.0,
                    'total_ms': total_ms,
                },
            }
        
        # Create hashes from query
        hash_started = perf_counter()
        if use_single_peaks:
            query_hashes, query_hash_data = create_single_peak_hashes(peaks)
            db_to_search = self.single_peak_db
        else:
            query_hashes, query_hash_data = create_hashes(peaks)
            db_to_search = self.database
        hash_generation_ms = (perf_counter() - hash_started) * 1000
        
        # Look up hashes in database and collect offsets
        offsets_by_song = {song: [] for song in self.song_list}
        total_matches = 0
        
        lookup_started = perf_counter()
        for query_hash, (_, query_time) in zip(query_hashes, query_hash_data):
            if query_hash in db_to_search:
                database_entries = db_to_search[query_hash]
                total_matches += len(database_entries)
                
                for db_song_name, db_time in database_entries:
                    # Genuine matches agree on database_time - query_time.
                    offset = float(db_time) - float(query_time)
                    offsets_by_song[db_song_name].append(offset)
        database_lookup_ms = (perf_counter() - lookup_started) * 1000
        
        # Create offset histogram for each song
        scoring_started = perf_counter()
        song_scores = {}
        
        for song_name in self.song_list:
            offsets = np.array(offsets_by_song[song_name])
            
            if len(offsets) == 0:
                song_scores[song_name] = 0
            else:
                # Use fixed half-second bins so scores are comparable between songs.
                aligned_offsets = np.round(offsets / 0.5) * 0.5
                song_scores[song_name] = Counter(aligned_offsets).most_common(1)[0][1]
        
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

        matched_song_offsets = (
            np.asarray(offsets_by_song[matched_song], dtype=float)
            if matched_song else np.array([])
        )
        scoring_ms = (perf_counter() - scoring_started) * 1000
        total_ms = (perf_counter() - total_started) * 1000
        
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
    'use_single_peaks': use_single_peaks,
    'duration': duration,
    'total_hash_matches': total_matches,
    'timings': {
        'audio_load_ms': load_ms,
        'spectrogram_ms': spectrogram_ms,
        'constellation_ms': constellation_ms,
        'hash_generation_ms': hash_generation_ms,
        'database_lookup_ms': database_lookup_ms,
        'scoring_ms': scoring_ms,
        'total_ms': total_ms,
    }
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
        for query_hash, (_, query_time) in zip(query_hashes, query_hash_data):
            if query_hash in self.database:
                for db_song, db_time in self.database[query_hash]:
                    if matched_song is None or db_song == matched_song:
                        offsets.append(float(db_time) - float(query_time))
        
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
