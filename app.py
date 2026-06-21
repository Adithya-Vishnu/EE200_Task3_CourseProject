"""
EE200 Q3B - Audio Fingerprinting Web Application

Interactive Streamlit app for identifying songs from audio clips.
Features:
- Single-clip mode: Identify individual songs with visualization
- Batch mode: Identify multiple songs and export results
- Real-time spectrogram and peak visualization
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import librosa
import librosa.display
import pandas as pd
from pathlib import Path
import tempfile
import os
import pickle
from io import BytesIO
import csv

# Import our modules
from fingerprint import (
    load_audio, compute_spectrogram, extract_peaks, 
    plot_constellation, N_FFT, HOP_LENGTH, PEAK_NEIGHBORHOOD, PEAK_AMP_MIN
)
from database_builder import SongDatabase
from matcher import SongMatcher


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="EE200: Audio Fingerprinting",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0f0f1e;
        color: #e0e0e0;
    }
    .stApp {
        background-color: #0f0f1e;
    }
    h1, h2, h3 {
        color: #00d9ff;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] button {
        color: #e0e0e0;
        background-color: #1a1a2e;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #00d9ff;
        color: #0f0f1e;
    }
    .match-success {
        background-color: #1a4d2e;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #2ecc71;
    }
    .match-failure {
        background-color: #4d1a1a;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #e74c3c;
    }
    .info-box {
        background-color: #1a2a4d;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #00d9ff;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_resource
def load_database(db_path="song_database.pkl"):
    """Load the song database (cached for performance)."""
    db = SongDatabase()
    if os.path.exists(db_path):
        db.load(db_path)
        return db
    else:
        st.warning(f"⚠️ Database not found at {db_path}")
        return None


def plot_spectrogram_streamlit(S_db, freqs, times, sr=22050):
    """Create a spectrogram plot for display."""
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#0f0f1e')
    ax.set_facecolor('#1a1a2e')
    
    img = librosa.display.specshow(S_db, sr=sr, hop_length=HOP_LENGTH,
                                    x_axis='time', y_axis='hz', ax=ax, cmap='magma')
    ax.set_ylim([0, 5000])
    ax.set_title('Spectrogram: Time-Frequency Analysis', fontsize=14, fontweight='bold', color='#00d9ff')
    ax.set_xlabel('Time (s)', fontsize=12, color='white')
    ax.set_ylabel('Frequency (Hz)', fontsize=12, color='white')
    
    # Styling
    ax.tick_params(colors='white')
    cbar = fig.colorbar(img, ax=ax, label='Amplitude (dB)')
    cbar.ax.tick_params(colors='white')
    cbar.set_label('Amplitude (dB)', color='white')
    
    return fig


def plot_constellation_streamlit(S_db, freqs, times, peaks, song_name="Query"):
    """Create constellation map plot."""
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#0f0f1e')
    ax.set_facecolor('#1a1a2e')
    
    img = librosa.display.specshow(S_db, sr=22050, hop_length=HOP_LENGTH,
                                    x_axis='time', y_axis='hz', ax=ax, cmap='magma', alpha=0.9)
    
    # Overlay peaks as constellation
    if peaks:
        peak_freqs = [p[0] for p in peaks]
        peak_times = [p[1] for p in peaks]
        ax.scatter(peak_times, peak_freqs, color='#00d9ff', s=100, alpha=0.9, 
                  edgecolors='white', linewidth=1.5, label='Peaks')
    
    ax.set_ylim([0, 5000])
    ax.set_title(f'Constellation Map: {song_name} ({len(peaks)} peaks)', 
                fontsize=14, fontweight='bold', color='#00d9ff')
    ax.set_xlabel('Time (s)', fontsize=12, color='white')
    ax.set_ylabel('Frequency (Hz)', fontsize=12, color='white')
    
    # Styling
    ax.tick_params(colors='white')
    cbar = fig.colorbar(img, ax=ax, label='Amplitude (dB)')
    cbar.ax.tick_params(colors='white')
    cbar.set_label('Amplitude (dB)', color='white')
    ax.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
    
    return fig


def plot_offset_histogram_streamlit(offsets, matched_song_name, confidence):
    """Create offset histogram plot."""
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#0f0f1e')
    ax.set_facecolor('#1a1a2e')
    
    if len(offsets) > 0:
        ax.hist(offsets, bins=50, color='#FF8C42', edgecolor='white', alpha=0.85, linewidth=1.5)
        ax.set_ylabel('Number of Matching Hashes', fontsize=12, color='white')
        ax.set_xlabel('Time Offset (seconds)', fontsize=12, color='white')
        ax.set_title(f'Offset Histogram: {matched_song_name}\nPeak = {int(confidence)} matching hashes (Confidence)',
                    fontsize=14, fontweight='bold', color='#00d9ff')
        ax.grid(True, alpha=0.2, color='white', linestyle='--')
        ax.set_ylim([0, None])
    else:
        ax.text(0.5, 0.5, 'No matches found', ha='center', va='center',
               transform=ax.transAxes, fontsize=16, color='#e74c3c', fontweight='bold')
        ax.set_title('Offset Histogram: No Matches', fontsize=14, fontweight='bold', color='#00d9ff')
    
    ax.tick_params(colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    
    return fig


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("🎵 **EE200: Audio Fingerprinting**", unsafe_allow_html=True)
    st.markdown("**Signals, Systems & Networks • Project Demo**")
    st.markdown("---")
    
    st.markdown("""
    Index a library of songs as spectrogram fingerprints, then identify any short clip against it.
    """)
    
    # Load database
    db = load_database("song_database.pkl")
    
    if db is None:
        st.error("❌ Database not loaded. Please ensure song_database.pkl exists in the app directory.")
        st.info("To build a database:\n1. Place audio files in a folder\n2. Run the database builder")
        return
    
    # Show database stats
    with st.sidebar:
        st.header("📊 Database Status")
        stats = db.get_statistics()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Songs Indexed", stats['num_songs'])
            st.metric("Avg Hashes/Song", f"{stats['avg_hashes_per_song']:.0f}")
        with col2:
            st.metric("Unique Hashes", f"{stats['num_unique_hashes']:,}")
            st.metric("Avg Peaks/Song", f"{stats['avg_peaks_per_song']:.0f}")
        
        st.markdown("---")
        st.markdown("**Indexed Songs:**")
        for i, song in enumerate(sorted(db.song_list), 1):
            st.caption(f"{i}. {song}")
    
    # Main tabs
    tab1, tab2 = st.tabs(["🎤 IDENTIFY (Single Clip)", "📦 BATCH (Multiple Clips)"])
    
    # ========================================================================
    # TAB 1: SINGLE CLIP MODE
    # ========================================================================
    
    with tab1:
        st.header("Identify a Clip")
        st.markdown("Upload a short audio clip and we'll identify the song from your database.")
        
        col_upload, col_examples = st.columns([1, 1])
        
        with col_upload:
            st.subheader("Upload Audio")
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=['mp3', 'wav', 'flac', 'ogg', 'm4a'],
                help="Max 200MB per file • Supports WAV, MP3, FLAC, OGG, M4A"
            )
        
        with col_examples:
            st.subheader("Sample Clips")
            st.markdown("No file? You can test with any song from the database:")
            # Create a dropdown to select sample songs for quick testing
            use_sample_db = st.checkbox(
                "Use sample from database",
                key="use_sample_db"
            )

            if use_sample_db:
                selected_song = st.selectbox(
                    "Choose a song:",
                    sorted(db.song_list)
                )
        
        if uploaded_file is not None or (use_sample_db and 'selected_song' in locals()):
            # Save uploaded file temporarily
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    query_path = tmp_file.name
            else:
                # Find the sample song file (would need actual path - for demo purposes)
                base_dir = os.path.dirname(os.path.abspath(__file__))

                query_path = os.path.join(
                    base_dir,
                    "EE200 Project Song Database",
                    selected_song + ".mp3"
                )
                st.write(query_path)
                st.write(os.path.exists(query_path))

                print("QUERY PATH =", query_path)
                print("EXISTS =", os.path.exists(query_path))
                            
            if query_path:
                st.markdown("---")
                
                # Match the query
                with st.spinner("🔍 Fingerprinting and matching..."):
                    matcher = SongMatcher(db)
                    result = matcher.match_query(query_path, use_single_peaks=False, confidence_threshold=10)
                
                # STEP 1: Feature Extraction
                st.subheader("STEP 1 • Feature Extraction")
                st.markdown("The clip was converted into a time-frequency map (left); from that rich image, only the **{} most prominent peaks** were kept (right). Discarding amplitude and phase makes the fingerprint robust to EQ, volume changes, and noise.".format(
                    len(result['peaks'])))
                
                col_spec, col_const = st.columns(2)
                
                with col_spec:
                    st.markdown("**Spectrogram**")
                    if result['S_db'] is not None:
                        fig_spec = plot_spectrogram_streamlit(result['S_db'], result['freqs'], result['times'])
                        st.pyplot(fig_spec, use_container_width=True)
                
                with col_const:
                    st.markdown("**Constellation Map**")
                    if result['S_db'] is not None:
                        fig_const = plot_constellation_streamlit(result['S_db'], result['freqs'], 
                                                                result['times'], result['peaks'])
                        st.pyplot(fig_const, use_container_width=True)
                
                st.markdown("---")
                
                # STEP 2: Database Search
                st.subheader("STEP 2 • Database Search")
                
                if result['matched_song']:
                    st.markdown(f"The {len(result['query_hashes'])} fingerprint hashes were looked up against every indexed track. Below is the full fingerprint of {result['matched_song']} reconstructed from the database. Each dot-arrow pair shows a match anchor. The highlighted window is exactly where the clip occurs in the song.")
                else:
                    st.markdown(f"Searched {len(result['query_hashes'])} hashes against {stats['num_songs']} songs in the database.")
                
                # STEP 3: The Proof
                st.subheader("STEP 3 • The Proof")
                st.markdown("Every matched hash votes for a time offset (database frame minus query frame). Chance matches scatter votes randomly, forming a flat noise floor. A genuine match makes them converge: **{} hashes agreed on a single offset.** That spike cannot be a coincidence.".format(
                    int(result['confidence']) if result['confidence'] > 0 else 0))
                
                if result['S_db'] is not None and result['matched_song']:
                    fig_hist = plot_offset_histogram_streamlit(result['offsets'], result['matched_song'], result['confidence'])
                    st.pyplot(fig_hist, use_container_width=True)
                
                st.markdown("---")
                
                # RESULT
                st.subheader("🎵 RESULT")
                
                if result['matched_song']:
                    st.markdown(f"""
                    <div class="match-success">
                    <h2 style="color: #2ecc71; margin: 0;">✅ MATCH FOUND</h2>
                    <h1 style="color: #00d9ff; margin: 10px 0 0 0;">{result['matched_song']}</h1>
                    <p style="font-size: 16px; color: #e0e0e0; margin: 10px 0 0 0;">
                    <strong>Confidence Score:</strong> {int(result['confidence'])} matching hashes<br>
                    <strong>Query Peaks:</strong> {len(result['peaks'])}<br>
                    <strong>Method:</strong> Paired-Hash Fingerprinting
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show top 5 candidates
                    st.markdown("**Top Candidate Scores:**")
                    sorted_scores = sorted(result['song_scores'].items(), key=lambda x: x[1], reverse=True)
                    
                    for i, (song, score) in enumerate(sorted_scores[:5], 1):
                        if song == result['matched_song']:
                            st.success(f"{i}. **{song}**: {int(score)} ⭐ (MATCH)")
                        else:
                            st.caption(f"{i}. {song}: {int(score)}")
                
                else:
                    st.markdown(f"""
                    <div class="match-failure">
                    <h2 style="color: #e74c3c; margin: 0;">❌ NO MATCH</h2>
                    <p style="font-size: 14px; color: #e0e0e0; margin: 10px 0 0 0;">
                    The fingerprint did not match any song in the database with sufficient confidence.
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.info("**Why might this happen?**\n- Song not in database\n- Clip is too short or poor quality\n- Heavy background noise\n- Very different arrangement or remix")
                
                # Cleanup
                os.unlink(query_path)
    
    # ========================================================================
    # TAB 2: BATCH MODE
    # ========================================================================
    
    with tab2:
        st.header("Batch Identification")
        st.markdown("Upload multiple audio clips and get a CSV file with predictions.")
        
        uploaded_files = st.file_uploader(
            "Choose audio files",
            type=['mp3', 'wav', 'flac', 'ogg', 'm4a'],
            accept_multiple_files=True,
            help="Select multiple files to process in batch"
        )
        
        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} files selected**")
            
            if st.button("🚀 Identify All", key="batch_button"):
                matcher = SongMatcher(db)
                results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.container()
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    # Update progress
                    progress = (idx + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    
                    # Extract filename without extension for the prediction label
                    filename_base = Path(uploaded_file.name).stem
                    
                    status_text.text(f"Processing: {uploaded_file.name} ({idx + 1}/{len(uploaded_files)})")
                    
                    # Save temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Match
                        result = matcher.match_query(tmp_path, use_single_peaks=False, confidence_threshold=10)
                        
                        # Store result
                        prediction = result['matched_song'] if result['matched_song'] else "NO_MATCH"
                        results.append({
                            'filename': filename_base,
                            'prediction': prediction
                        })
                    
                    except Exception as e:
                        results.append({
                            'filename': filename_base,
                            'prediction': 'ERROR'
                        })
                    
                    finally:
                        os.unlink(tmp_path)
                
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                st.markdown("---")
                st.subheader("Results")
                
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True)
                
                # Download CSV
                csv_buffer = BytesIO()
                df_results.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                
                st.download_button(
                    label="📥 Download results.csv",
                    data=csv_buffer,
                    file_name="results.csv",
                    mime="text/csv",
                    key="download_csv"
                )
                
                # Statistics
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    matches = (df_results['prediction'] != 'NO_MATCH').sum()
                    st.metric("Matched", f"{matches}/{len(results)}")
                with col2:
                    no_matches = (df_results['prediction'] == 'NO_MATCH').sum()
                    st.metric("No Match", no_matches)
                with col3:
                    success_rate = (matches / len(results) * 100) if len(results) > 0 else 0
                    st.metric("Success Rate", f"{success_rate:.1f}%")
        
        else:
            st.info("👆 Select files above to get started")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <strong>How it works:</strong><br>
    1. Audio clip → Spectrogram (time-frequency representation)<br>
    2. Extract constellation of prominent peaks<br>
    3. Create hashes from peak pairs (freq₁, freq₂, Δt)<br>
    4. Look up hashes in database → collect time offsets<br>
    5. Find offset with spike (many hashes align) = match!<br><br>
    <strong>Why paired peaks?</strong> Single frequencies are common across many songs. 
    But (freq1, freq2, time_gap) is rare and distinctive.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()