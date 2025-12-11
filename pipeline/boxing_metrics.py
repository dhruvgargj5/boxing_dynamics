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
                shoulder_position=self.get_shoulder_joints_over_time(
                    input.position
                ),
                hip_position=self.get_hip_joints_over_time(
                    input.position
                ),
                hip_rotation_velocity_magnitude=self.calculate_hip_rotation(
                    input
                ),
                shoulder_rotation_velocity_magnitude=self.calculate_shoulder_rotation(
                    input
                ),
                center_of_mass=self.calculate_com(input.position),
            )
        else:
            raise NotImplementedError

    def get_shoulder_joints_over_time(self, position: np.ndarray):
        return np.stack(
            [
                position[:, PoseLandmark.LEFT_SHOULDER],
                position[:, PoseLandmark.RIGHT_SHOULDER],
            ],
            axis=-1,
        )

    def get_hip_joints_over_time(self, position: np.ndarray):
        return np.stack(
            [
                position[:, PoseLandmark.LEFT_HIP],
                position[:, PoseLandmark.RIGHT_HIP],
            ],
            axis=-1,
        )

    def calculate_com(self, position: np.ndarray):
        # https://biomechanist.net/center-of-mass/
        mass = {
            "head": 7.6,
            "torso": 44.2,
            "upper_arm": 3.2,
            "lower_arm": 1.7,
            "hand": 0.9,
            "thigh": 11.9,
            "calf": 4.6,
            "foot": 2,
        }
        pos = {
            "head": lambda: position[:, PoseLandmark.NOSE],
            "torso": lambda: get_torso(),
            "upper_arm_R": lambda: get_midpoint(
                PoseLandmark.RIGHT_SHOULDER,
                PoseLandmark.RIGHT_ELBOW,
                0.436,
                0.564,
            ),
            "upper_arm_L": lambda: get_midpoint(
                PoseLandmark.LEFT_SHOULDER,
                PoseLandmark.LEFT_ELBOW,
                0.436,
                0.564,
            ),
            "lower_arm_R": lambda: get_midpoint(
                PoseLandmark.RIGHT_ELBOW,
                PoseLandmark.RIGHT_WRIST,
                0.43,
                0.57,
            ),
            "lower_arm_L": lambda: get_midpoint(
                PoseLandmark.LEFT_ELBOW,
                PoseLandmark.LEFT_WRIST,
                0.43,
                0.57,
            ),
            "hand_R": lambda: position[:, PoseLandmark.RIGHT_WRIST],
            "hand_L": lambda: position[:, PoseLandmark.LEFT_WRIST],
            "thigh_R": lambda: get_midpoint(
                PoseLandmark.RIGHT_HIP,
                PoseLandmark.RIGHT_KNEE,
                0.43,
                0.57,
            ),
            "thigh_L": lambda: get_midpoint(
                PoseLandmark.LEFT_HIP,
                PoseLandmark.LEFT_KNEE,
                0.43,
                0.57,
            ),
            "calf_R": lambda: get_midpoint(
                PoseLandmark.RIGHT_KNEE,
                PoseLandmark.RIGHT_ANKLE,
                0.43,
                0.57,
            ),
            "calf_L": lambda: get_midpoint(
                PoseLandmark.LEFT_KNEE,
                PoseLandmark.LEFT_ANKLE,
                0.43,
                0.57,
            ),
            "foot_R": lambda: position[:, PoseLandmark.RIGHT_ANKLE],
            "foot_L": lambda: position[:, PoseLandmark.LEFT_ANKLE],
        }
        get_midpoint = lambda l, r, lw=1.0, rw=1.0: (
            position[:, l] * lw + position[:, r] * rw
        ) / (lw + rw)

        def get_torso():
            top_torso = get_midpoint(
                PoseLandmark.RIGHT_SHOULDER,
                PoseLandmark.LEFT_SHOULDER,
            )
            bottom_torso = get_midpoint(
                PoseLandmark.RIGHT_HIP, PoseLandmark.LEFT_HIP
            )
            return (top_torso + bottom_torso) / 2

        com_f = lambda key: mass[key] * pos[key]()
        com_rl = lambda key, side: mass[key] * pos[f"{key}_{side}"]()
        com = (
            com_f("head")
            + com_f("torso")
            + com_rl("upper_arm", "R")
            + com_rl("upper_arm", "L")
            + com_rl("lower_arm", "R")
            + com_rl("lower_arm", "L")
            + com_rl("hand", "R")
            + com_rl("hand", "L")
            + com_rl("thigh", "R")
            + com_rl("thigh", "L")
            + com_rl("calf", "R")
            + com_rl("calf", "L")
            + com_rl("foot", "R")
            + com_rl("foot", "L")
        )

        return com

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
