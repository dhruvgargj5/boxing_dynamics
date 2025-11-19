#!/usr/bin/env python3

import click
import logging
from pathlib import Path

from pipeline.pipeline import (
    VideoConfiguration,
    LandmarkingStageInput,
)
from pipeline.video_loader import VideoLoader
from pipeline.landmarking import ExtractHumanPoseLandmarks
from pipeline.kinematics_extractor import (
    ExtractWorldLandmarkLinearKinematics,
    ExtractJointAngularKinematics,
)
from pipeline.boxing_metrics import CalculateBoxingMetrics
from pipeline.video_ouput import FuseVideoAndBoxingMetrics
from mediapipe.python.solutions.pose import PoseLandmark
import mediapipe as mp
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerOptions
from mediapipe.tasks.python import BaseOptions

def kinematics_callback(ctx, param, value):
    mapping = {
        "left_knee": PoseLandmark.LEFT_KNEE,
        "right_knee": PoseLandmark.RIGHT_KNEE,
        "left_elbow" :PoseLandmark.LEFT_ELBOW,
        "right_elbow" :PoseLandmark.RIGHT_ELBOW,
    }
    return mapping[value.lower()]

@click.command()
@click.argument(
    "video_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--debug-logging",
    is_flag=True,
    help="Enable DEBUG logging",
    default=False,
)
@click.option(
    "--scale-factor",
    type=float,
    default=None,
    help="Optional scale factor for resizing video frames (e.g. 0.5).",
)
@click.option("--lite", "model_fidelity", flag_value="lite", default='lite', help="Use lite MediaPipe model (default).")
@click.option("--heavy", "model_fidelity", flag_value="heavy", help="heavy model.")
@click.option(
    "--kinematics",
    type=click.Choice(["left_knee", "right_knee", "left_elbow", "right_elbow"], case_sensitive=False),
    required=False,
    default=None,
    help="Which joint kinematics to plot.",
    callback=lambda ctx, param, value: PoseLandmark[value.upper()]
)
def main(video_path: Path, debug_logging: bool, scale_factor: float, model_fidelity, kinematics):
    """Run the BoxingDynamics pipeline on a specified video path."""
    log_level = logging.DEBUG if debug_logging else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    logging.info("Starting BoxingDynamics pipeline")
    logging.info(f"Video Path: {video_path}")

    # Build video configuration dynamically
    video_config = VideoConfiguration(
        name=video_path.stem,
        path=video_path,
        scale_factor=scale_factor,
    )

    # --- Run pipeline stages ---
    video_data = VideoLoader().execute(video_config)
    model_asset_path = None
    match model_fidelity:
        case 'heavy':
            model_asset_path = "assets/pose_landmarker_heavy.task"
        case _:
            model_asset_path = "assets/pose_landmarker_lite.task"
    

    landmarkers = ExtractHumanPoseLandmarks().execute(
        LandmarkingStageInput(
            video_data,
            PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=model_asset_path
                ),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                output_segmentation_masks=False,
            ),
        )
    )

    linear_kinematics = ExtractWorldLandmarkLinearKinematics().execute(landmarkers)
    boxing_metrics = CalculateBoxingMetrics().execute(linear_kinematics)

    video_fuser = FuseVideoAndBoxingMetrics()

    if kinematics is not None:
        logging.info(f"Outputting kinematics")
        joint_angle_kinematics = ExtractJointAngularKinematics().execute(linear_kinematics)
        video_fuser.PlotJointAngularKinematics((video_data, joint_angle_kinematics), kinematics)
        return
    
    output_path = video_fuser.execute(
        (video_data, boxing_metrics)
    )
    logging.info(f"Output video saved to: {output_path}")
    logging.info("Finished BoxingDynamics pipeline")


if __name__ == "__main__":
    main()
