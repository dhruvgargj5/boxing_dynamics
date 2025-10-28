import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarkerResult,
)

from pipeline.pipeline import (
    StageBase,
    WorldLandmarkLinearKinematicVariables,
    AngularKinematicVariables,
    JointAngularKinematicVariables,
)

from typing import List


class ExtractWorldLandmarkLinearKinematics(
    StageBase[
        List[PoseLandmarkerResult],
        WorldLandmarkLinearKinematicVariables,
    ]
):
    def execute(
        self, input: List[PoseLandmarkerResult]
    ) -> WorldLandmarkLinearKinematicVariables:
        positions = mediapipe_human_frame_landmark_to_position_array(
            input
        )
        return WorldLandmarkLinearKinematicVariables(positions)


class ExtractJointAngularKinematics(
    StageBase[
        WorldLandmarkLinearKinematicVariables,
        JointAngularKinematicVariables,
    ]
):
    def execute(
        self, input: WorldLandmarkLinearKinematicVariables
    ) -> JointAngularKinematicVariables:
        joint_3d_angles = np.zeros_like(input.position)
        return JointAngularKinematicVariables(
            AngularKinematicVariables(joint_3d_angles)
        )


def mediapipe_human_frame_landmark_to_position_array(
    time_series_landmarks: List[PoseLandmarkerResult],
):
    num_frames = len(time_series_landmarks)
    num_landmarks = len(
        time_series_landmarks[0].pose_world_landmarks[0]
    )

    positions = np.zeros(
        (num_frames, num_landmarks, 3), dtype=np.float32
    )

    for frame_idx, landmarker_result in enumerate(
        time_series_landmarks
    ):
        for landmark_idx, pose_landmark in enumerate(
            landmarker_result.pose_world_landmarks[0]
        ):
            positions[frame_idx, landmark_idx, :] = np.array(
                [pose_landmark.x, pose_landmark.y, pose_landmark.z]
            )

    return positions
