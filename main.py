#!/usr/bin/env python3

import click
import logging
from pathlib import Path

from pipeline.pipeline import (
    VideoConfiguration,
    LandmarkingStageInput,
    AddArrowsStageInput,
)
from pipeline.video_loader import VideoLoader
from pipeline.landmarking import ExtractHumanPoseLandmarks
from pipeline.kinematics_extractor import (
    ExtractWorldLandmarkLinearKinematics,
    ExtractJointAngularKinematics,
)
from pipeline.boxing_metrics import CalculateBoxingMetrics
from pipeline.video_ouput import FuseVideoAndBoxingMetrics
from pipeline.add_arrows import AddArrowsToLegs

import mediapipe as mp
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerOptions
from mediapipe.tasks.python import BaseOptions

# adding command line call options
@click.command()
@click.argument(
    "video_path", 
    type=str)
@click.option("--name", type=str, default=None,
              help="Optional display name for the video (e.g., 'goodRightHook').")
@click.option(
    "--scale-factor",
    type=float,
    default=None,
    help="Optional scale factor for resizing video frames (e.g. 0.5).",
)
@click.option("--lite", "model_fidelity", flag_value="lite", default='lite', help="Use lite MediaPipe model (default).")
@click.option("--heavy", "model_fidelity", flag_value="heavy", help="heavy model.")
@click.option(
    "--debug-logging",
    is_flag=True,
    help="Enable DEBUG logging",
    default=False,
)


# main function : Running the pipeline
def main(video_path: str, name: str, debug_logging: bool, scale_factor: float, model_fidelity):
    """Run the BoxingDynamics pipeline on a specified video path."""
    log_level = logging.DEBUG if debug_logging else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    logging.info("Starting BoxingDynamics pipeline")
    logging.info(f"Video Path: {video_path}")

    # Stage 1️⃣: Load video
    video_config = VideoConfiguration(
        name=name,                  
        path=Path(video_path),
        scale_factor=scale_factor,
    )
    video_data = VideoLoader().execute(video_config)
    
    # Stage 2️⃣:  Select model (lite/heavy) and extract landmarks
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

    # Stage3️⃣: Compute linear kinematics 
    linear_kinematics = ExtractWorldLandmarkLinearKinematics().execute(landmarkers)
    
    # Stage 4️⃣: Compute joint angular kinematics
    joint_angle_kinematics = ExtractJointAngularKinematics().execute(linear_kinematics)
    
    # Stage 5️⃣: Calculate Relevant Boxing Metrics
    boxing_metrics = CalculateBoxingMetrics().execute(linear_kinematics)

    # Stage 6️⃣: Add (growning/shrinking) force arrows to the orginal video
    video_data_w_arrows, output_path = AddArrowsToLegs().execute(AddArrowsStageInput(
                                                                                    video_data, 
                                                                                    landmarkers, 
                                                                                    save_video=True))
    
    

    # Stage 7️⃣: Fuse the input video and boxing metrics into one output video
    output_path = FuseVideoAndBoxingMetrics().execute((video_data, boxing_metrics))
    
    # uncomment the line below to have video with arrows on the left and boxing metric plots on the right
    # output_path = FuseVideoAndBoxingMetrics().execute((video_data_w_arrows, boxing_metrics))

    logging.info(f"Output video with boxing feedback is saved to: {output_path}")
    logging.info(f"Saving video with arrows to: {output_path}")

    logging.info("Finished BoxingDynamics pipeline")

if __name__ == "__main__":
    main()