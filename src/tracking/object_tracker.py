import json
from pathlib import Path

from .ball_tracker import BallTracker
from .hoop_tracker import HoopTracker
from .player_tracker import PlayerTracker


class ObjectTracker:
    def __init__(self):
        project_root = Path(__file__).resolve().parents[2]

        thresholds_path = (
            project_root
            / "config"
            / "selected_thresholds.json"
        )

        with open(
            thresholds_path,
            "r",
            encoding="utf-8",
        ) as file:
            thresholds = json.load(file)

        self.ball_tracker = BallTracker(
            confidence_threshold=thresholds["basketball"]
        )

        self.hoop_tracker = HoopTracker(
            confidence_threshold=thresholds["hoop"]
        )

        self.player_tracker = PlayerTracker(
            confidence_threshold=thresholds["player"]
        )

        print("Tracking thresholds:")
        print(f"  Basketball: {thresholds['basketball']}")
        print(f"  Hoop: {thresholds['hoop']}")
        print(f"  Player: {thresholds['player']}")

    def track(self, detection_result, frame):
        return {
            "basketball": self.ball_tracker.track(
                detection_result,
                frame,
            ),
            "hoop": self.hoop_tracker.track(
                detection_result,
                frame,
            ),
            "player": self.player_tracker.track(
                detection_result,
                frame,
            ),
        }

    def get_stats(self):
        return {
            "basketball": self.ball_tracker.get_stats(),
            "hoop": self.hoop_tracker.get_stats(),
            "player": self.player_tracker.get_stats(),
        }

    def reset(self):
        self.ball_tracker.reset()
        self.hoop_tracker.reset()
        self.player_tracker.reset()