"""
EE200 Q3A - Song Database Builder

This module creates a fingerprint database from all songs in the database.
Each song is processed through the full fingerprinting pipeline and
the resulting hashes are stored in a dictionary for fast lookup.

Database structure:
{
    hash_tuple: [(song_name, anchor_time), (song_name, anchor_time), ...],
    ...
}
"""

import os
import pickle
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from fingerprint import (
    load_audio, compute_spectrogram, extract_peaks, 
    create_hashes, create_single_peak_hashes,
    N_FFT, HOP_LENGTH, PEAK_NEIGHBORHOOD, PEAK_AMP_MIN
)


class SongDatabase:
    """
    A fingerprint database for song identification.
    """
    
    def __init__(self):
        """Initialize empty database."""
        self.database = {}  # hash -> [(song_name, time), ...]
        self.single_peak_db = {}  # For single-peak comparison
        self.song_list = []  # List of songs in database
        self.peak_count_per_song = {}  # For statistics
        self.hash_count_per_song = {}
    
    def add_song(self, song_path, song_name=None):
        """
        Add a song to the database.
        
        Parameters
        ----------
        song_path : str
            Path to audio file
        song_name : str, optional
            Display name. If None, extracted from filename
        """
        if song_name is None:
            song_name = Path(song_path).stem  # Filename without extension
        
        print(f"  Indexing: {song_name}...", end=" ", flush=True)
        
        try:
            # Load audio
            y, sr, duration = load_audio(song_path)
            
            # Compute spectrogram
            S_db, freqs, times = compute_spectrogram(y, sr)
            
            # Extract peaks
            peaks, _ = extract_peaks(S_db, freqs, times, 
                                     neighborhood=PEAK_NEIGHBORHOOD,
                                     amp_min=PEAK_AMP_MIN)
            
            # Create hashes from peaks
            hashes, hash_data = create_hashes(peaks)
            
            # Add to database
            for hash_tuple, anchor_time in hash_data:
                if hash_tuple not in self.database:
                    self.database[hash_tuple] = []
                self.database[hash_tuple].append((song_name, anchor_time))
            
            # Also create single-peak hashes for comparison
            single_hashes, single_hash_data = create_single_peak_hashes(peaks)
            for single_hash, anchor_time in single_hash_data:
                if single_hash not in self.single_peak_db:
                    self.single_peak_db[single_hash] = []
                self.single_peak_db[single_hash].append((song_name, anchor_time))
            
            # Store statistics
            self.song_list.append(song_name)
            self.peak_count_per_song[song_name] = len(peaks)
            self.hash_count_per_song[song_name] = len(hashes)
            
            print(f"✓ ({len(peaks)} peaks, {len(hashes)} hashes, {duration:.1f}s)")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
    
    def batch_index(self, song_dir):
        """
        Index all songs in a directory.
        
        Parameters
        ----------
        song_dir : str
            Directory containing audio files
        
        Returns
        -------
        n_songs : int
            Number of songs indexed
        """
        song_dir = Path(song_dir)
        
        # Find all audio files
        audio_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
        song_files = []
        
        for ext in audio_extensions:
            song_files.extend(song_dir.glob(f'*{ext}'))
        
        song_files.sort()
        
        print(f"\n📀 Indexing {len(song_files)} songs from: {song_dir}\n")
        
        for song_file in song_files:
            self.add_song(str(song_file))
        
        print(f"\n Database complete!")
        print(f"   - Total songs: {len(self.song_list)}")
        print(f"   - Total unique hashes: {len(self.database)}")
        print(f"   - Avg hashes per song: {sum(self.hash_count_per_song.values()) / len(self.song_list):.0f}")
        
        return len(self.song_list)
    
    def save(self, filepath):
        """
        Save database to disk using pickle.
        
        Parameters
        ----------
        filepath : str
            Path to save database file
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'database': self.database,
                'single_peak_db': self.single_peak_db,
                'song_list': self.song_list,
                'peak_count_per_song': self.peak_count_per_song,
                'hash_count_per_song': self.hash_count_per_song
            }, f)
        print(f"💾 Database saved to: {filepath}")
    
    def load(self, filepath):
        """
        Load database from disk.
        
        Parameters
        ----------
        filepath : str
            Path to database file
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.database = data['database']
            self.single_peak_db = data['single_peak_db']
            self.song_list = data['song_list']
            self.peak_count_per_song = data['peak_count_per_song']
            self.hash_count_per_song = data['hash_count_per_song']
        print(f" Database loaded from: {filepath}")
        print(f"   - {len(self.song_list)} songs")
        print(f"   - {len(self.database)} unique hashes")
    
    def get_statistics(self):
        """
        Get database statistics.
        
        Returns
        -------
        stats : dict
            Various statistics about the database
        """
        total_hashes = sum(len(v) for v in self.database.values())
        
        return {
            'num_songs': len(self.song_list),
            'num_unique_hashes': len(self.database),
            'total_hash_occurrences': total_hashes,
            'avg_hashes_per_song': np.mean(list(self.hash_count_per_song.values())),
            'avg_peaks_per_song': np.mean(list(self.peak_count_per_song.values())),
        }
    
    def print_statistics(self):
        """Print formatted statistics."""
        stats = self.get_statistics()
        print("\n" + "="*60)
        print("DATABASE STATISTICS")
        print("="*60)
        print(f"Songs indexed:           {stats['num_songs']}")
        print(f"Unique hashes:           {stats['num_unique_hashes']:,}")
        print(f"Total hash entries:      {stats['total_hash_occurrences']:,}")
        print(f"Avg hashes per song:     {stats['avg_hashes_per_song']:.0f}")
        print(f"Avg peaks per song:      {stats['avg_peaks_per_song']:.0f}")
        print("="*60 + "\n")


def build_database(song_directory, output_file="song_database.pkl"):
    """
    Convenience function to build database from a directory of songs.
    
    Parameters
    ----------
    song_directory : str
        Path to directory containing songs
    output_file : str
        Where to save the database
    
    Returns
    -------
    db : SongDatabase
        The built database
    """
    db = SongDatabase()
    db.batch_index(song_directory)
    db.print_statistics()
    db.save(output_file)
    return db


if __name__ == "__main__":
    # Example usage
    song_dir = "EE200 Project Song Database"
    
    if os.path.exists(song_dir):
        db = build_database(song_dir, "song_database.pkl")
    else:
        print(f"Directory not found: {song_dir}")
