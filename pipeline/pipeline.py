from pathlib import Path
from dataclasses import dataclass
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarkerOptions,
)
import numpy as np


from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, NamedTuple, Optional
import logging


InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class StageBase(ABC, Generic[InputType, OutputType]):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def execute(self, input: InputType) -> OutputType:
        raise NotImplementedError


@dataclass
class VideoConfiguration:
    name: str
    path: Path
    scale_factor: Optional[float] = None


class Frame(NamedTuple):
    frame: np.ndarray
    timestamp_ms: int


@dataclass
class VideoData:
    frames: List[Frame]
    fps: float
    config: VideoConfiguration


class LandmarkingStageInput(NamedTuple):
    video_data: VideoData
    landmarking_options: PoseLandmarkerOptions


@dataclass
class WorldLandmarkLinearKinematicVariables:
    # (Time/frame, landmark, linear pos/vel/accel (3))
    position: np.ndarray
    velocity: Optional[np.ndarray] = None
    acceleration: Optional[np.ndarray] = None


@dataclass
class AngularKinematicVariables:
    # (Time/frame, joint, pos/vel/accel (1))
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


@dataclass
class JointAnatomicalAngularKinematicVariables:
    transverse: AngularKinematicVariables
    sagittal: AngularKinematicVariables
    frontal: AngularKinematicVariables


@dataclass
class JointAngularKinematicVariables:
    joint_3d_angular_kinematics: AngularKinematicVariables
    joint_planar_angular_kinematics: Optional[
        JointAnatomicalAngularKinematicVariables
    ] = None

@dataclass
class BoxingPunchMetrics:
    right_wrist_punching_velocity_magnitude: np.ndarray
    left_wrist_punching_velocity_magnitude: np.ndarray
    hip_rotation_velocity_magnitude: Optional[np.ndarray] = None
    shoulder_rotation_velocity_magnitude: Optional[np.ndarray] = None
