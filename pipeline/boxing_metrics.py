import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarkerResult,
)
from mediapipe.python.solutions.pose import PoseLandmark

from pipeline.pipeline import (
    StageBase,
    WorldLandmarkLinearKinematicVariables,
    AngularKinematicVariables,
    JointAngularKinematicVariables,
    BoxingPunchMetrics,
)

from utils.joints import JOINTS, Joint

from typing import List, Tuple


class CalculateBoxingMetrics(
    StageBase[
        WorldLandmarkLinearKinematicVariables, BoxingPunchMetrics
    ]
):
    def execute(
        self, input: WorldLandmarkLinearKinematicVariables
    ) -> BoxingPunchMetrics:
        if input.velocity is not None:
            return BoxingPunchMetrics(
                right_wrist_punching_velocity_magnitude=np.linalg.norm(
                    input.velocity[:, PoseLandmark.RIGHT_WRIST],
                    axis=1,
                ),
                left_wrist_punching_velocity_magnitude=np.linalg.norm(
                    input.velocity[:, PoseLandmark.LEFT_WRIST],
                    axis=1,
                ),
                hip_rotation_velocity_magnitude=self.calculate_hip_rotation(
                    input
                ),
                shoulder_rotation_velocity_magnitude=self.calculate_shoulder_rotation(
                    input
                ),
            )
        else:
            raise NotImplementedError

    def calculate_hip_rotation(
        self, input: WorldLandmarkLinearKinematicVariables
    ):
        return self._calculate_rotation_of_body_segment(
            input, PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP
        )

    def calculate_shoulder_rotation(
        self, input: WorldLandmarkLinearKinematicVariables
    ):
        return self._calculate_rotation_of_body_segment(
            input,
            PoseLandmark.LEFT_SHOULDER,
            PoseLandmark.RIGHT_SHOULDER,
        )

    def _calculate_rotation_of_body_segment(
        self,
        input: WorldLandmarkLinearKinematicVariables,
        landmark_a: PoseLandmark,
        landmark_b: PoseLandmark,
    ):
        position = (
            input.position[:, landmark_a]
            - input.position[:, landmark_b]
        )
        velocity = (
            input.velocity[:, landmark_a]
            - input.velocity[:, landmark_b]
        )
        pos_vel_cross = np.cross(position, velocity)
        pos_norm_sq = np.sum(position**2, axis=1, keepdims=True)
        omega = pos_vel_cross / pos_norm_sq
        return np.linalg.norm(omega, axis=1)
