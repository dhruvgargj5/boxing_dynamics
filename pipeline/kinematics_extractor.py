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
)

from utils.joints import JOINTS, Joint

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
        self.logger.info("Starting ExtractWorldLandmarkLinearKinematics stage")
        positions = mediapipe_human_frame_landmark_to_position_array(
            input
        )
        velocities = np.gradient(positions, axis=0)
        self.logger.info("Finishing ExtractWorldLandmarkLinearKinematics stage")
        return WorldLandmarkLinearKinematicVariables(
            positions, velocity=velocities
        )


class ExtractJointAngularKinematics(
    StageBase[
        WorldLandmarkLinearKinematicVariables,
        JointAngularKinematicVariables,
    ]
):
    def execute(
        self, input: WorldLandmarkLinearKinematicVariables
    ) -> JointAngularKinematicVariables:

        desired_joints = [
            PoseLandmark.LEFT_ELBOW,
            PoseLandmark.RIGHT_ELBOW,
            PoseLandmark.LEFT_KNEE,
            PoseLandmark.RIGHT_KNEE,
        ]
        self.logger.info(f"Desired joints: {desired_joints}")
        joint_3d_angles = np.zeros(
            (input.position.shape[0], len(desired_joints))
        )

        joint_3d_angular_vel = np.zeros_like(joint_3d_angles)
        joint_3d_angular_accel = np.zeros_like(joint_3d_angles)

        for joint in desired_joints:
            proximal_limb_vector = (
                input.position[:, JOINTS[joint].parent_landmark]
                - input.position[:, JOINTS[joint].joint_landmark]
            )
            distal_limb_vector = (
                input.position[:, JOINTS[joint].joint_landmark]
                - input.position[:, JOINTS[joint].child_landmark]
            )
            angular_positions = (
                calculate_nominal_joint_angles_from_time_series(
                    proximal_limb_vector, distal_limb_vector
                )
            )
            angular_vel = np.gradient(angular_positions)
            angular_accel = np.gradient(angular_vel)

            joint_3d_angles[:, JOINTS[joint].index] = angular_positions
            joint_3d_angular_vel[:, JOINTS[joint].index] = angular_vel
            joint_3d_angular_accel[:, JOINTS[joint].index] = angular_accel
        return JointAngularKinematicVariables(
            AngularKinematicVariables(
                joint_3d_angles,
                joint_3d_angular_vel,
                joint_3d_angular_accel,
            )
        )


def calculate_nominal_joint_angles_from_time_series(
    proximal_vec_over_time, distal_vec_over_time
):
    dot_product = np.einsum(
        "ij,ij->i", proximal_vec_over_time, distal_vec_over_time
    )
    similarity = dot_product / (
        np.linalg.norm(proximal_vec_over_time, axis=1)
        * np.linalg.norm(distal_vec_over_time, axis=1)
    )
    return np.arccos(similarity)


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
        if landmarker_result.pose_world_landmarks:
            for landmark_idx, pose_landmark in enumerate(
                landmarker_result.pose_world_landmarks[0]
            ):
                positions[frame_idx, landmark_idx, :] = np.array(
                    [pose_landmark.x, pose_landmark.y, pose_landmark.z]
                )

    return positions
