from pathlib import Path

from ultralytics import YOLO


class ObjectDetector:
    def __init__(self):
        project_root = Path(__file__).resolve().parent.parent
        model_path = (
            project_root
            / "results"
            / "yolo26n"
            / "baseline"
            / "weights"
            / "best.pt"
        )

        self.model = YOLO(model_path)

        print(f"Model loaded: {model_path}")
        print(f"Classes: {self.model.names}")

    def detect(self, frame):
        results = self.model.predict(
            source=frame,
            conf=0.05,
            verbose=False,
        )

        return results[0]