import cv2


class VideoProcessor:
    def __init__(self, video_path, detector, tracker):
        self.video_path = video_path
        self.detector = detector
        self.tracker = tracker

    def run(self):
        video = cv2.VideoCapture(self.video_path)

        if not video.isOpened():
            raise FileNotFoundError(
                f"Could not open video: {self.video_path}"
            )

        cv2.namedWindow(
            "Basketball Tracking",
            cv2.WINDOW_NORMAL,
        )
        cv2.resizeWindow(
            "Basketball Tracking",
            960,
            540,
        )

        colors = {
            "basketball": (0, 0, 255),
            "hoop": (255, 0, 0),
            "player": (255, 255, 0),
        }

        while True:
            success, frame = video.read()

            if not success:
                break

            detection_result = self.detector.detect(frame)

            tracked_objects = self.tracker.track(
                detection_result,
                frame,
            )

            annotated_frame = frame.copy()

            for class_name, tracks in tracked_objects.items():
                color = colors[class_name]

                for track in tracks:
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        track_id,
                        score,
                        class_id,
                        _,
                    ) = track

                    x1, y1, x2, y2 = map(
                        int,
                        (x1, y1, x2, y2),
                    )

                    label = (
                        f"{class_name} | "
                        f"ID: {int(track_id)} | "
                        f"Conf: {score:.2f}"
                    )

                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        3,
                    )

                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4,
                        color,
                        7,
                    )

            cv2.imshow(
                "Basketball Tracking",
                annotated_frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        video.release()
        print("\nTracking Summary")

        for class_name, stats in self.tracker.get_stats().items():
            print(f"\n{class_name}")

            for name, value in stats.items():
                print(f"  {name}: {value}")
        cv2.destroyAllWindows()