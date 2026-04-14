import time
from collections import defaultdict


class EventManager:
    def __init__(self, confirmation_frames=15, cooldown_seconds=10):
        self.confirmation_frames = confirmation_frames
        self.cooldown_seconds = cooldown_seconds

        self.violation_frame_counts = defaultdict(int)
        self.last_event_time = defaultdict(float)

    def update(self, current_violations):
        """
        current_violations: dict like
        {
            "no hat": 1,
            "no vest": 0,
            "no gloves": 2
        }
        """
        confirmed_events = []

        # Increase frame count for active violations
        for violation_name, count in current_violations.items():
            if count > 0:
                self.violation_frame_counts[violation_name] += 1
            else:
                self.violation_frame_counts[violation_name] = 0

        current_time = time.time()

        for violation_name, frame_count in self.violation_frame_counts.items():
            if frame_count >= self.confirmation_frames:
                time_since_last = current_time - self.last_event_time[violation_name]

                if time_since_last >= self.cooldown_seconds:
                    confirmed_events.append(violation_name)
                    self.last_event_time[violation_name] = current_time
                    self.violation_frame_counts[violation_name] = 0

        return confirmed_events