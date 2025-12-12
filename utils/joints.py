from mediapipe.python.solutions.pose import PoseLandmark

from dataclasses import dataclass


@dataclass
class Joint:
    index: int
    parent_landmark: PoseLandmark
    joint_landmark: PoseLandmark
    child_landmark: PoseLandmark

    def get_proximal_limb_landmark_indexes(self):
        return [self.parent_landmark.value, self.joint_landmark.value]

    def get_distal_limb_landmark_indexes(self):
        return [self.joint_landmark.value, self.child_landmark.value]

    def get_landmarks(self):
        return [
            self.parent_landmark,
            self.joint_landmark,
            self.child_landmark,
        ]


joint_definitions = [
    (
        PoseLandmark.LEFT_HIP,
        PoseLandmark.LEFT_KNEE,
        PoseLandmark.LEFT_ANKLE,
    ),
    (
        PoseLandmark.RIGHT_HIP,
        PoseLandmark.RIGHT_KNEE,
        PoseLandmark.RIGHT_ANKLE,
    ),
    (
        PoseLandmark.RIGHT_SHOULDER,
        PoseLandmark.RIGHT_ELBOW,
        PoseLandmark.RIGHT_WRIST,
    ),
    (
        PoseLandmark.LEFT_SHOULDER,
        PoseLandmark.LEFT_ELBOW,
        PoseLandmark.LEFT_WRIST,
    ),
]

JOINTS = {
    target_joint: Joint(ii, parent, target_joint, child)
    for ii, (parent, target_joint, child) in enumerate(
        joint_definitions
    )
}
