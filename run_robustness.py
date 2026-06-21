from database_builder import SongDatabase
from q3a_experiments import *

db = SongDatabase()
db.load("song_database.pkl")

song_file = "EE200 Project Song Database/Hey Jude.mp3"

noise_results = test_noise_robustness(db, song_file)
pitch_results = test_pitch_shift(db, song_file)
stretch_results = test_time_stretch(db, song_file)