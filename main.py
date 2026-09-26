from pathlib import Path

from src.detector import ObjectDetector
from src.tracking import ObjectTracker
from src.video_processor import VideoProcessor


def main():
    project_root = Path(__file__).resolve().parent
    video_path = (
        project_root
        / "videos"
        / "test_video2.mp4"
    )

    detector = ObjectDetector()
    tracker = ObjectTracker()

    processor = VideoProcessor(
        video_path=video_path,
        detector=detector,
        tracker=tracker,
    )

    processor.run()


if __name__ == "__main__":
    main()