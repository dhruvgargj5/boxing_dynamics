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

# ---------------------------------------------------------------------
# WRAPPER FUNCTION: For Hugging Face / Gradio / API use
# ---------------------------------------------------------------------
def process_video(
    video_path: str,
    name: str = None,
    scale_factor: float = None,
    model_fidelity: str = "lite",
    debug_logging: bool = False,
) -> str:
    """
    Process a video using the BoxingDynamics pipeline and return
    the *output video file path*.

    This wraps the CLI main() so HF Spaces or Gradio can call it directly.
    """
    return _run_pipeline(
        video_path=video_path,
        name=name,
        scale_factor=scale_factor,
        model_fidelity=model_fidelity,
        debug_logging=debug_logging,
    )


# ---------------------------------------------------------------------
# INTERNAL PIPELINE FUNCTION (shared by process_video and CLI)
# ---------------------------------------------------------------------
def _run_pipeline(
    video_path: str,
    name: str,
    scale_factor: float,
    model_fidelity: str,
    debug_logging: bool,
) -> str:

    log_level = logging.DEBUG if debug_logging else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    logging.info("Starting BoxingDynamics pipeline")
    logging.info(f"Video Path: {video_path}")

    # Stage 1: Load video
    video_config = VideoConfiguration(
        name=name,
        path=Path(video_path),
        scale_factor=scale_factor,
    )
    video_data = VideoLoader().execute(video_config)

    # Stage 2: Select model and extract landmarks
    match model_fidelity:
        case "heavy":
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

    # Stage 3: Linear kinematics
    linear_kinematics = (
        ExtractWorldLandmarkLinearKinematics().execute(landmarkers)
    )

    # Stage 4: Joint angles
    joint_angle_kinematics = ExtractJointAngularKinematics().execute(
        linear_kinematics
    )

    # Stage 5: Boxing metrics
    boxing_metrics = CalculateBoxingMetrics().execute(linear_kinematics)

    # Stage 6: Fuse video & metrics into output
    output_path = FuseVideoAndBoxingMetrics().execute(
        (video_data, boxing_metrics)
    )

    logging.info(f"Output video is saved to: {output_path}")
    logging.info("Finished BoxingDynamics pipeline")

    return str(output_path)


# ---------------------------------------------------------------------
# ORIGINAL CLICK-BASED CLI ENTRY POINT
# ---------------------------------------------------------------------
@click.command()
@click.argument("video_path", type=str)
@click.option(
    "--name",
    type=str,
    default=None,
    help="Optional display name for the video (e.g., 'goodRightHook').",
)
@click.option(
    "--scale-factor",
    type=float,
    default=None,
    help="Optional scale factor for resizing video frames (e.g. 0.5).",
)
@click.option("--lite", "model_fidelity", flag_value="lite", default="lite")
@click.option("--heavy", "model_fidelity", flag_value="heavy")
@click.option("--debug-logging", is_flag=True, default=False)
def main(video_path, name, debug_logging, scale_factor, model_fidelity):
    """CLI wrapper: python main.py file.mp4"""
    process_video(
        video_path,
        name=name,
        scale_factor=scale_factor,
        model_fidelity=model_fidelity,
        debug_logging=debug_logging,
    )


if __name__ == "__main__":
    main()
