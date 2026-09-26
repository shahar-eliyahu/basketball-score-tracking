from collections import defaultdict

from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml


class HoopTracker:
    def __init__(self, confidence_threshold):
        self.class_id = 1
        self.confidence_threshold = confidence_threshold

        config_path = check_yaml("bytetrack.yaml")
        config = IterableSimpleNamespace(**YAML.load(config_path))

        config.track_high_thresh = confidence_threshold
        config.new_track_thresh = confidence_threshold

        self.tracker = BYTETracker(args=config)

        self.frames_processed = 0
        self.frames_with_detections = 0
        self.total_detections = 0
        self.frames_with_detections_but_no_tracks = 0

        self.seen_track_ids = set()
        self.track_lengths = defaultdict(int)

    def track(self, detection_result, frame):
        self.frames_processed += 1

        boxes = detection_result.boxes
        hoop_boxes = boxes[
            boxes.cls == self.class_id
        ].cpu().numpy()

        detection_count = len(hoop_boxes)

        if detection_count > 0:
            self.frames_with_detections += 1
            self.total_detections += detection_count

        tracks = self.tracker.update(
            hoop_boxes,
            frame,
        )

        if detection_count > 0 and len(tracks) == 0:
            self.frames_with_detections_but_no_tracks += 1

        for track in tracks:
            track_id = int(track[4])

            self.seen_track_ids.add(track_id)
            self.track_lengths[track_id] += 1

        return tracks

    def get_stats(self):
        longest_track = max(
            self.track_lengths.values(),
            default=0,
        )

        return {
            "confidence_threshold": self.confidence_threshold,
            "frames": self.frames_processed,
            "frames_with_detections": self.frames_with_detections,
            "total_detections": self.total_detections,
            "unique_track_ids": len(self.seen_track_ids),
            "frames_with_detections_but_no_tracks":
                self.frames_with_detections_but_no_tracks,
            "longest_track": longest_track,
        }

    def reset(self):
        self.tracker.reset()

        self.frames_processed = 0
        self.frames_with_detections = 0
        self.total_detections = 0
        self.frames_with_detections_but_no_tracks = 0

        self.seen_track_ids.clear()
        self.track_lengths.clear()