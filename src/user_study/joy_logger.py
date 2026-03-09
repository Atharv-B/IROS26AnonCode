import pandas as pd
import os

class JoyLogger:
    def __init__(self):
        self.joy_events = []
        self.num_axes = -1
        self.num_buttons = -1

    def log_joy_event(self, ts, axes, buttons):
        self.num_axes = len(axes)
        self.num_buttons = len(buttons)

        row = [ts] + list(axes) + list(buttons)
        self.joy_events.append(row)

    def get_joy_events(self):
        return self.joy_events

    def save_joy_events(self, _dir):
        columns_raw = ["timestamp"] + [f"axis_{i}" for i in range(self.num_axes)] + [f"button_{i}" for i in range(self.num_buttons)]
        df = pd.DataFrame(self.joy_events, columns=columns_raw)
        df.to_csv(os.path.join(_dir, "joy.csv"), index=False)

