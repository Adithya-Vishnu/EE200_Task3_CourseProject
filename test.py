import soundfile as sf
from fingerprint import load_audio
from database_builder import SongDatabase
from matcher import SongMatcher

song = "EE200 Project Song Database/Hey Jude.mp3"

db = SongDatabase()
db.load("song_database.pkl")

matcher = SongMatcher(db)

# Original MP3
result = matcher.match_query(song)

print("ORIGINAL")
print("Matched:", result["matched_song"])
print("Confidence:", result["confidence"])

# Save exact WAV copy
y, sr, _ = load_audio(song)
sf.write("test.wav", y, sr)

# WAV version
result = matcher.match_query("test.wav")

print("\nWAV COPY")
print("Matched:", result["matched_song"])
print("Confidence:", result["confidence"])