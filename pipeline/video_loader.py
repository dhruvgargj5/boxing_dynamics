import cv2
import numpy as np

from pipeline.pipeline import (
    StageBase,
    VideoConfiguration,
    VideoData,
    Frame,
)


class VideoLoader(StageBase[VideoConfiguration, VideoData]):
    def execute(self, input: VideoConfiguration) -> VideoData:
        self.logger.info("Starting VideoLoader stage")
        cap = cv2.VideoCapture(str(input.path))

        frames = []
        msg = "Loading frames from video"
        if input.scale_factor is not None:
            msg += f", downscaling by {input.scale_factor}"
        self.logger.info(msg)

        while cap.isOpened():
            successful, frame = cap.read()

            if not successful:
                self.logger.info(
                    "Can't read frame, breaking out of video read"
                )
                break
            if input.scale_factor is not None:
                frame = cv2.resize(
                    frame,
                    dsize=None,
                    fx=input.scale_factor,
                    fy=input.scale_factor,
                )
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(
                Frame(frame, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
            )
        _fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        self.logger.info(
            f"Loaded {len(frames)} frames from {str(input.path)}"
        )
        self.logger.info("Finished VideoLoader stage")

        return VideoData(
            frames=frames,
            fps=_fps,
            config=input,
        )
