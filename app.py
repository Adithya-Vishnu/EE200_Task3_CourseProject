"""EE200 audio fingerprinting demo built with Streamlit."""

from collections import Counter
from html import escape
from pathlib import Path
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import pandas as pd
import streamlit as st

from database_builder import SongDatabase
from fingerprint import HOP_LENGTH, N_FFT
from matcher import SongMatcher


APP_DIR = Path(__file__).resolve().parent
SONG_DIR = APP_DIR / "EE200 Project Song Database"
DATABASE_PATH = APP_DIR / "song_database.pkl"
SUPPORTED_TYPES = ["wav", "mp3", "flac", "ogg", "m4a"]
SAMPLE_NAMES = [
    "While My Guitar Gently Weeps",
    "Bohemian Rhapsody",
    "The Long And Winding Road",
    "Two Of Us",
    "Within You Without You",
]


st.set_page_config(
    page_title="EE200 · Audio Fingerprinting",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --cyan:#23d5e6; --panel:#171b2d; --muted:#9ba7bd; }
      .stApp { background:#0b0e18; color:#edf2f7; }
      [data-testid="stSidebar"] { background:#111525; }
      h1, h2, h3 { color:#ecf8ff; letter-spacing:-0.02em; }
      .eyebrow { color:#23d5e6; font-size:.78rem; font-weight:800;
                 letter-spacing:.14em; text-transform:uppercase; }
      .hero { padding:.7rem 0 1.2rem; }
      .hero h1 { margin:.1rem 0; font-size:2.45rem; }
      .hero p { color:#aeb9ca; max-width:760px; margin:.2rem 0; }
      .stTabs [data-baseweb="tab-list"] { gap:.35rem; }
      .stTabs [data-baseweb="tab"] { background:#14182a; border-radius:10px 10px 0 0;
                                      padding:.75rem 1.35rem; color:#b9c3d4; }
      .stTabs [aria-selected="true"] { background:#173b49 !important; color:#58e4f0 !important; }
      div[data-testid="stMetric"] { background:#151a2b; border:1px solid #252c43;
                                    padding:.8rem; border-radius:12px; }
      div[data-testid="stVerticalBlockBorderWrapper"] { border-color:#293149; background:#121625; }
      .result-ok { background:linear-gradient(135deg,#12352e,#122537); border:1px solid #28cb91;
                   border-radius:14px; padding:1.35rem 1.5rem; }
      .result-no { background:#351c25; border:1px solid #df5f76;
                   border-radius:14px; padding:1.25rem 1.5rem; }
      .result-ok h2,.result-no h2 { margin:.1rem 0; }
      .result-ok p,.result-no p { color:#cbd5e1; margin:.35rem 0 0; }
      .step-note { color:#aeb9ca; margin-top:-.35rem; }
      .sample-label { color:#eff6ff; font-weight:700; padding-top:.45rem; }
      .sample-sub { color:#8390a6; font-size:.82rem; }
      .footer { color:#76839a; text-align:center; padding:1.5rem 0 .5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_database(db_path: str, modified_ns: int):
    """Load the database once, invalidating the cache when its file changes."""
    del modified_ns
    database = SongDatabase()
    database.load(db_path)
    return database


def dark_figure(width=12, height=5):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("#0b0e18")
    ax.set_facecolor("#14182a")
    ax.tick_params(colors="#b9c3d4")
    for spine in ax.spines.values():
        spine.set_color("#384058")
    return fig, ax


def finish_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, color="#58e4f0", weight="bold", pad=12)
    ax.set_xlabel(xlabel, color="#dce5f2")
    ax.set_ylabel(ylabel, color="#dce5f2")
    ax.grid(alpha=.12, color="white", linestyle="--")


def plot_spectrogram(S_db):
    fig, ax = dark_figure()
    image = librosa.display.specshow(
        S_db, sr=22050, hop_length=HOP_LENGTH,
        x_axis="time", y_axis="hz", cmap="magma", ax=ax,
    )
    ax.set_ylim(0, 5000)
    finish_axes(ax, "Spectrogram", "Time (seconds)", "Frequency (Hz)")
    colorbar = fig.colorbar(image, ax=ax, pad=.01)
    colorbar.set_label("Power (dB)", color="#dce5f2")
    colorbar.ax.tick_params(colors="#b9c3d4")
    fig.tight_layout()
    return fig


def plot_constellation(S_db, peaks):
    fig, ax = dark_figure()
    librosa.display.specshow(
        S_db, sr=22050, hop_length=HOP_LENGTH,
        x_axis="time", y_axis="hz", cmap="magma", alpha=.55, ax=ax,
    )
    visible = [point for point in peaks if point[0] <= 5000]
    if visible:
        ax.scatter(
            [point[1] for point in visible], [point[0] for point in visible],
            s=16, c="#43e6ed", edgecolors="white", linewidths=.25, alpha=.85,
        )
    ax.set_ylim(0, 5000)
    finish_axes(ax, f"Constellation · {len(peaks):,} peaks", "Time (seconds)", "Frequency (Hz)")
    fig.tight_layout()
    return fig


def plot_spectral_scatter(S_db):
    """Render the raw high-energy spectral points in frame/bin coordinates."""
    visible = S_db[:301]
    threshold = np.percentile(visible, 98.5)
    freq_bins, time_frames = np.where(visible >= threshold)
    amplitudes = visible[freq_bins, time_frames]
    if len(time_frames) > 14000:
        sample = np.linspace(0, len(time_frames) - 1, 14000, dtype=int)
        freq_bins, time_frames, amplitudes = (
            freq_bins[sample], time_frames[sample], amplitudes[sample]
        )
    fig, ax = dark_figure()
    ax.scatter(time_frames, freq_bins, c=amplitudes, cmap="autumn", s=5, alpha=.58)
    finish_axes(ax, "Step 1 · Spectral energy map", "Time (frames)", "Frequency bin")
    fig.tight_layout()
    return fig


def plot_constellation_scatter(peaks):
    points = [point for point in peaks if point[0] <= 6500]
    fig, ax = dark_figure()
    if points:
        frame = np.asarray([point[1] * 22050 / HOP_LENGTH for point in points])
        freq_bin = np.asarray([point[0] / (22050 / N_FFT) for point in points])
        ax.scatter(frame, freq_bin, c="#ffb23f", s=10, alpha=.75)
    finish_axes(ax, f"Step 2 · Sparse constellation ({len(peaks):,} anchors)",
                "Time (frames)", "Frequency bin")
    fig.tight_layout()
    return fig


def plot_offset_histogram(offsets, song_name, confidence):
    fig, ax = dark_figure()
    if len(offsets):
        aligned = np.round(np.asarray(offsets) / .5) * .5
        counts = Counter(aligned)
        ordered = sorted(counts.items())
        x = np.asarray([item[0] for item in ordered])
        y = np.asarray([item[1] for item in ordered])
        colors = np.full(len(y), "#e98a32", dtype=object)
        colors[int(np.argmax(y))] = "#42e2e8"
        ax.bar(x, y, width=.42, color=colors, alpha=.9)
        finish_axes(
            ax, f"The alignment spike · {song_name}",
            "Database time − query time (seconds)", "Matching-hash votes",
        )
        peak_index = int(np.argmax(y))
        ax.annotate(
            f"{int(confidence):,} agreeing hashes",
            (x[peak_index], y[peak_index]), xytext=(10, 16),
            textcoords="offset points", color="#76f1f2", weight="bold",
            arrowprops={"arrowstyle": "->", "color": "#76f1f2"},
        )
    else:
        ax.text(.5, .5, "No aligned hashes", transform=ax.transAxes,
                ha="center", va="center", color="#e27789", size=15)
        finish_axes(ax, "The alignment spike", "Time offset", "Votes")
    fig.tight_layout()
    return fig


def show_figure(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def locate_song(song_name):
    for path in SONG_DIR.iterdir() if SONG_DIR.exists() else []:
        if path.is_file() and path.stem == song_name:
            return path
    return None


def sample_tracks(database):
    names = [name for name in SAMPLE_NAMES if name in database.song_list]
    names.extend(name for name in sorted(database.song_list) if name not in names)
    tracks = []
    for name in names:
        path = locate_song(name)
        if path:
            tracks.append((name, path))
        if len(tracks) == 5:
            break
    return tracks


def render_samples(database, key_prefix):
    """Show five native audio players and return a clicked sample request."""
    request = None
    st.markdown("#### Or try a sample")
    st.caption("Each player includes play/pause, seek, timestamp, duration, and volume controls.")
    for index, (name, path) in enumerate(sample_tracks(database), 1):
        with st.container(border=True):
            label_col, player_col, button_col = st.columns([1.3, 4.8, .7], vertical_alignment="center")
            with label_col:
                st.markdown(
                    f'<div class="sample-label">sample{index}</div>'
                    f'<div class="sample-sub">{escape(name)}</div>',
                    unsafe_allow_html=True,
                )
            with player_col:
                st.audio(str(path), format=f"audio/{path.suffix.lstrip('.')}")
            with button_col:
                if st.button("Try", key=f"{key_prefix}_sample_{index}", type="primary"):
                    request = {"path": str(path), "name": f"sample{index} · {name}"}
    return request


def save_upload(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower() or ".wav"
    temporary = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        temporary.write(uploaded_file.getbuffer())
        return temporary.name
    finally:
        temporary.close()


def run_match(database, path):
    return SongMatcher(database).match_query(
        path, use_single_peaks=False, confidence_threshold=10, max_duration=30,
    )


def timing_frame(result):
    timings = result.get("timings", {})
    labels = [
        ("Audio load", "audio_load_ms"),
        ("Spectrogram", "spectrogram_ms"),
        ("Constellation", "constellation_ms"),
        ("Hash generation", "hash_generation_ms"),
        ("Database lookup", "database_lookup_ms"),
        ("Scoring / ranking", "scoring_ms"),
    ]
    return pd.DataFrame(
        [{"Stage": label, "Time (ms)": round(float(timings.get(key, 0)), 2)}
         for label, key in labels]
    )


def render_match_result(result, query_name, detailed=True):
    if result.get("S_db") is not None and detailed:
        st.markdown("### 1 · Feature extraction")
        st.markdown(
            f'<p class="step-note">The first 30 seconds produced '
            f'<b>{len(result.get("peaks", [])):,}</b> stable anchor points.</p>',
            unsafe_allow_html=True,
        )
        left, right = st.columns(2)
        with left:
            show_figure(plot_spectrogram(result["S_db"]))
        with right:
            show_figure(plot_constellation(result["S_db"], result.get("peaks", [])))

    sorted_scores = sorted(
        result.get("song_scores", {}).items(), key=lambda item: item[1], reverse=True,
    )
    runner_up = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
    matched_song = result.get("matched_song")
    confidence = int(result.get("confidence", 0))

    if detailed:
        st.markdown("### 2 · Hash lookup")
        st.markdown(
            f'<p class="step-note"><b>{len(result.get("query_hashes", [])):,}</b> query hashes '
            f'produced <b>{result.get("total_hash_matches", 0):,}</b> database hits. Each hit voted '
            f'for a song and an alignment offset.</p>', unsafe_allow_html=True,
        )
        st.markdown("### 3 · The proof")
        st.markdown(
            '<p class="step-note">Chance matches scatter across offsets; a genuine excerpt creates '
            'a narrow, dominant spike.</p>', unsafe_allow_html=True,
        )
        show_figure(plot_offset_histogram(
            result.get("offsets", np.array([])), matched_song or "No match", confidence,
        ))

    st.markdown("### Result")
    if matched_song:
        st.markdown(
            f'<div class="result-ok"><div class="eyebrow">Match found</div>'
            f'<h2>{escape(matched_song)}</h2><p>Query: {escape(query_name)} · '
            f'Cluster score <b>{confidence:,}</b> · '
            f'<b>{max(0, confidence - int(runner_up)):,}</b> votes ahead of the runner-up</p></div>',
            unsafe_allow_html=True,
        )
        dominance = confidence / max(1, confidence + int(runner_up))
        st.progress(min(1.0, dominance), text=f"Top-match dominance · {dominance:.1%}")
    else:
        reason = result.get("reason", "No candidate exceeded the confidence threshold.")
        st.markdown(
            f'<div class="result-no"><div class="eyebrow">No reliable match</div>'
            f'<h2>Unknown clip</h2><p>{escape(reason)}</p></div>',
            unsafe_allow_html=True,
        )

    score_col, timing_col = st.columns([1, 1])
    with score_col:
        st.markdown("#### Candidate ranking")
        ranking = pd.DataFrame(
            [{"Rank": i, "Song": song, "Cluster score": int(score)}
             for i, (song, score) in enumerate(sorted_scores[:5], 1)]
        )
        st.dataframe(ranking, hide_index=True, use_container_width=True)
    with timing_col:
        st.markdown("#### Processing time")
        st.dataframe(timing_frame(result), hide_index=True, use_container_width=True)
        st.metric("Total", f'{result.get("timings", {}).get("total_ms", 0):,.1f} ms')


def build_library_previews(database, max_points=260):
    signature = (len(database.database), tuple(database.song_list))
    if st.session_state.get("library_preview_signature") == signature:
        return st.session_state["library_preview_points"]

    previews = {song: [] for song in database.song_list}
    complete = set()
    for fingerprint_hash, entries in database.database.items():
        if len(fingerprint_hash) < 3:
            continue
        first_freq, second_freq, delta = fingerprint_hash
        for song, anchor_time in entries:
            if song in complete:
                continue
            if first_freq <= 5000:
                previews[song].append((float(anchor_time), float(first_freq)))
            if len(previews[song]) < max_points and second_freq <= 5000:
                previews[song].append((float(anchor_time) + float(delta), float(second_freq)))
            if len(previews[song]) >= max_points:
                complete.add(song)
        if len(complete) == len(previews):
            break

    st.session_state["library_preview_signature"] = signature
    st.session_state["library_preview_points"] = previews
    return previews


def plot_library_thumbnail(points):
    fig, ax = dark_figure(width=5, height=2.25)
    if points:
        values = np.asarray(points)
        ax.scatter(values[:, 0], values[:, 1], c=values[:, 1], cmap="plasma", s=5, alpha=.8)
    ax.set_ylim(0, 5000)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    fig.tight_layout(pad=.2)
    return fig


def render_library(database):
    st.header("Fingerprint library")
    st.caption("Song indexing is managed by the admin. Drop a clip in the Identify tab to test the library.")
    search = st.text_input("Search indexed songs", placeholder="Type a song title…")
    songs = [song for song in sorted(database.song_list) if search.lower() in song.lower()]
    previews = build_library_previews(database)

    page_size = 12
    page_count = max(1, int(np.ceil(len(songs) / page_size)))
    page = st.selectbox("Page", range(1, page_count + 1), format_func=lambda value: f"{value} of {page_count}")
    visible = songs[(page - 1) * page_size:page * page_size]
    if not visible:
        st.info("No indexed songs match that search.")
        return

    columns = st.columns(3)
    for index, song in enumerate(visible):
        with columns[index % 3]:
            with st.container(border=True):
                show_figure(plot_library_thumbnail(previews.get(song, [])))
                st.markdown(f"**{song}**")
                hashes = database.hash_count_per_song.get(song, 0)
                peaks = database.peak_count_per_song.get(song, 0)
                st.caption(f"{hashes:,} hashes · {peaks:,} constellation points")


def render_identify(database):
    st.header("Identify a clip")
    st.caption("Upload up to 200 MB in WAV, MP3, FLAC, OGG, or M4A. The first 30 seconds are analyzed.")
    uploaded = st.file_uploader("Choose an audio clip", type=SUPPORTED_TYPES, key="identify_upload")
    request = None
    if uploaded and st.button("Identify uploaded clip", type="primary", use_container_width=True):
        temp_path = save_upload(uploaded)
        request = {"path": temp_path, "name": uploaded.name, "temporary": True}

    sample_request = render_samples(database, "identify")
    if sample_request:
        request = sample_request

    if request:
        try:
            with st.spinner("Building the query fingerprint and searching the library…"):
                result = run_match(database, request["path"])
            st.session_state["identify_result"] = (result, request["name"])
        except Exception as error:
            st.error(f"Could not process this clip: {error}")
        finally:
            if request.get("temporary") and os.path.exists(request["path"]):
                os.unlink(request["path"])

    if "identify_result" in st.session_state:
        st.divider()
        render_match_result(*st.session_state["identify_result"], detailed=True)


def run_batch(database, uploaded_files):
    rows = []
    detail = None
    progress = st.progress(0, text="Preparing batch…")
    matcher = SongMatcher(database)
    for index, uploaded in enumerate(uploaded_files, 1):
        progress.progress((index - 1) / len(uploaded_files), text=f"Processing {uploaded.name}")
        path = save_upload(uploaded)
        try:
            result = matcher.match_query(path, confidence_threshold=10, max_duration=30)
            prediction = result.get("matched_song") or "NO_MATCH"
            rows.append({
                "filename": Path(uploaded.name).stem,
                "prediction": prediction,
                "cluster_score": int(result.get("confidence", 0)),
                "processing_ms": round(result.get("timings", {}).get("total_ms", 0), 2),
            })
            if detail is None or result.get("confidence", 0) > detail[0].get("confidence", 0):
                detail = (result, uploaded.name)
        except Exception as error:
            rows.append({
                "filename": Path(uploaded.name).stem,
                "prediction": "ERROR",
                "cluster_score": 0,
                "processing_ms": 0,
                "error": str(error),
            })
        finally:
            if os.path.exists(path):
                os.unlink(path)
    progress.progress(1.0, text="Batch complete")
    return pd.DataFrame(rows), detail


def render_batch_visualization(result, query_name):
    st.markdown("### Algorithm visualization")
    st.caption(f"Detailed pipeline for {query_name}")
    left, right = st.columns(2)
    with left:
        show_figure(plot_spectral_scatter(result["S_db"]))
    with right:
        show_figure(plot_constellation_scatter(result.get("peaks", [])))

    st.markdown("#### Step 3 · Hash generation")
    hash_col, lookup_col, score_col = st.columns(3)
    hash_col.metric("Query hashes", f'{len(result.get("query_hashes", [])):,}')
    lookup_col.metric("Database hits", f'{result.get("total_hash_matches", 0):,}')
    score_col.metric("Alignment votes", f'{int(result.get("confidence", 0)):,}')
    st.caption("Anchor pairs are encoded as (frequency₁, frequency₂, Δtime) and looked up in O(1) average time.")

    st.markdown("#### Step 4–5 · Database lookup and alignment spike")
    show_figure(plot_offset_histogram(
        result.get("offsets", np.array([])), result.get("matched_song") or "No match",
        result.get("confidence", 0),
    ))
    render_match_result(result, query_name, detailed=False)


def render_batch(database):
    st.header("Batch identification")
    st.caption("Process several clips, inspect the strongest alignment, and export a CSV report.")
    uploaded_files = st.file_uploader(
        "Choose multiple audio clips", type=SUPPORTED_TYPES,
        accept_multiple_files=True, key="batch_upload",
    )
    if uploaded_files:
        st.write(f"{len(uploaded_files)} file(s) ready")
        if st.button("Identify all", type="primary", use_container_width=True):
            frame, detail = run_batch(database, uploaded_files)
            st.session_state["batch_frame"] = frame
            st.session_state["batch_detail"] = detail

    with st.expander("Or process one of the five samples"):
        sample_request = render_samples(database, "batch")
    if sample_request:
        with st.spinner("Processing sample…"):
            result = run_match(database, sample_request["path"])
        st.session_state["batch_frame"] = pd.DataFrame([{
            "filename": sample_request["name"],
            "prediction": result.get("matched_song") or "NO_MATCH",
            "cluster_score": int(result.get("confidence", 0)),
            "processing_ms": round(result.get("timings", {}).get("total_ms", 0), 2),
        }])
        st.session_state["batch_detail"] = (result, sample_request["name"])

    frame = st.session_state.get("batch_frame")
    if frame is not None:
        st.divider()
        st.markdown("### Batch results")
        st.dataframe(frame, hide_index=True, use_container_width=True)
        csv_data = frame.to_csv(index=False).encode("utf-8")
        st.download_button("Download results.csv", csv_data, "results.csv", "text/csv")
        valid = frame[~frame["prediction"].isin(["NO_MATCH", "ERROR"])]
        one, two, three = st.columns(3)
        one.metric("Matched", f"{len(valid)}/{len(frame)}")
        two.metric("No match", int((frame["prediction"] == "NO_MATCH").sum()))
        three.metric("Success rate", f"{(100 * len(valid) / max(1, len(frame))):.1f}%")

    detail = st.session_state.get("batch_detail")
    if detail and detail[0].get("S_db") is not None:
        render_batch_visualization(*detail)


def main():
    st.markdown(
        '<div class="hero"><div class="eyebrow">EE200 · Signals, Systems & Networks</div>'
        '<h1>Audio Fingerprinting Lab</h1>'
        '<p>Index songs as spectral fingerprints, identify unknown excerpts, and inspect the '
        'alignment spike that proves a match.</p></div>',
        unsafe_allow_html=True,
    )

    if not DATABASE_PATH.exists():
        st.error("song_database.pkl was not found. Build the database before starting the app.")
        return
    try:
        database = load_database(str(DATABASE_PATH), DATABASE_PATH.stat().st_mtime_ns)
    except Exception as error:
        st.error(f"The fingerprint database could not be loaded: {error}")
        return

    stats = database.get_statistics()
    with st.sidebar:
        st.markdown("### Database status")
        left, right = st.columns(2)
        left.metric("Songs", stats["num_songs"])
        right.metric("Unique hashes", f'{stats["num_unique_hashes"]:,}')
        left.metric("Avg hashes", f'{stats["avg_hashes_per_song"]:,.0f}')
        right.metric("Avg peaks", f'{stats["avg_peaks_per_song"]:,.0f}')
        st.caption(f'Fingerprint entries: {stats["total_hash_occurrences"]:,}')
        st.divider()
        st.markdown("**Indexed songs**")
        for song in sorted(database.song_list):
            st.caption(song)

    library_tab, identify_tab, batch_tab = st.tabs(["LIBRARY", "IDENTIFY", "BATCH"])
    with library_tab:
        render_library(database)
    with identify_tab:
        render_identify(database)
    with batch_tab:
        render_batch(database)

    st.markdown(
        '<div class="footer">Spectrogram → constellation → paired hashes → database votes → alignment spike</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
