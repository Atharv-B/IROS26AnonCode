import json
import os

class GoalLogger:
    def __init__(self):
        self.goal_events = []

    def log_goal_event(self, ts, goal):
        row = [ts] + [{i: list(g) for i, g in enumerate(goal)}]
        # Check to see if the goals have changed
        if self.goal_events and self.goal_events[-1][1] == row[1]:
            # Skip logging if the last event is the same
            return
        self.goal_events.append(row)

    def get_goal_events(self):
        return self.goal_events

    def save_goal_events(self, _dir):
        events = {g[0] : g[1] for g in self.goal_events}
        # print(events)
        # Save as json
        with open(os.path.join(_dir, "goals.json"), "w") as f:
            json.dump(events, f, indent=4)
