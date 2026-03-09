import pandas as pd
import os

class AudioLogger:
    def __init__(self):
        self.audio_events = []

    def log_audio_event(self, ts, audio_msg):
        row = [ts, audio_msg]
        self.audio_events.append(row)

    def get_audio_events(self):
        return self.audio_events

    def save_audio_events(self, _dir):
        columns_raw = ["timestamp", "audio_message"]
        df = pd.DataFrame(self.audio_events, columns=columns_raw)
        df.to_csv(os.path.join(_dir, "audio_feedback.csv"), index=False)

