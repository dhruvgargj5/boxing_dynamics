import mediapipe as mp
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarkerResult,
)

from pipeline.pipeline import StageBase, LandmarkingStageInput

from typing import List

class ExtractHumanPoseLandmarks(
    StageBase[
        LandmarkingStageInput,
        List[PoseLandmarkerResult],
    ]
):
    def execute(
        self, input: LandmarkingStageInput
    ) -> List[PoseLandmarkerResult]:
        self.logger.info("Starting ExtractHumanPoseLandmarks stage")
        landmarker = (
            mp.tasks.vision.PoseLandmarker.create_from_options(
                input.landmarking_options
            )
        )
        landmarker_results = []
        for frame in input.video_data.frames:
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=frame.frame
            )
            pose_landmarker_result = landmarker.detect_for_video(
                mp_image, frame.timestamp_ms
            )
            landmarker_results.append(pose_landmarker_result)
        self.logger.info("Finished ExtractHumanPoseLandmarks stage")
        return landmarker_results
