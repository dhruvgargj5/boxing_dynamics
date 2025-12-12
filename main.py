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
from mediapipe.python.solutions.pose import PoseLandmark
from pipeline.add_arrows import AddArrowsToLegs

import mediapipe as mp
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarkerOptions,
)
from mediapipe.tasks.python import BaseOptions

# ---------------------------------------------------------------------
# WRAPPER FUNCTION: For Hugging Face / Gradio / API use
# ---------------------------------------------------------------------
def process_video(
    video_path: Path,
    output_dir: Path,
    no_metrics: bool,
    debug_logging: bool,
    scale_factor: float,
    model_fidelity,
    angular_kinematics_joints,
    linear_kinematics_joints,
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
    video_path: Path,
    output_dir: Path,
    no_metrics: bool,
    debug_logging: bool,
    scale_factor: float,
    model_fidelity,
    angular_kinematics_joints,
    linear_kinematics_joints,
) -> str:
    """Run the BoxingDynamics pipeline on a specified video path."""
    log_level = logging.DEBUG if debug_logging else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    logging.info("Starting BoxingDynamics pipeline")
    logging.info(f"Video Path: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Made output directory {output_dir}")

    # Build video configuration dynamically
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

    linear_kinematics = (
        ExtractWorldLandmarkLinearKinematics().execute(landmarkers)
    )
    boxing_metrics = CalculateBoxingMetrics().execute(
        linear_kinematics
    )

    video_fuser = FuseVideoAndBoxingMetrics()
    if linear_kinematics_joints:
        for joint_name in linear_kinematics_joints:
            joint = PoseLandmark[joint_name.upper()]
            logging.info(
                f"Outputting linear kinematics for {joint.name}"
            )
    if angular_kinematics_joints:
        for joint_name in angular_kinematics_joints:
            joint = PoseLandmark[joint_name.upper()]
            logging.info(
                f"Outputting angular kinematics for {joint.name}"
            )
            joint_angle_kinematics = (
                ExtractJointAngularKinematics().execute(
                    linear_kinematics
                )
            )
            kinematics_anim = video_fuser.PlotJointAngularKinematics(
                (video_data, joint_angle_kinematics, landmarkers),
                joint,
            )
            kinematics_out_path = (
                output_dir / f"{joint.name}_angular_kinematics.MP4"
            )
            kinematics_anim.save(
                kinematics_out_path,
                writer="ffmpeg",
                fps=video_data.fps,
            )
            logging.info(
                f"Kinematics video saved to: {kinematics_out_path}"
            )
    if not no_metrics:
        animation = video_fuser.execute((video_data, boxing_metrics))
        metrics_out_path = output_dir / "metrics.MP4"
        animation.save(
            metrics_out_path, writer="ffmpeg", fps=video_data.fps
        )
        logging.info(f"Metrics video saved to: {metrics_out_path}")
    logging.info("Finished BoxingDynamics pipeline")

    return str(output_path)


# ---------------------------------------------------------------------
# ORIGINAL CLICK-BASED CLI ENTRY POINT
# ---------------------------------------------------------------------


@click.command()
@click.argument(
    "video_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.argument(
    "output_dir",
    type=click.Path(path_type=Path, dir_okay=True, file_okay=False),
)
@click.option(
    "--no-metrics",
    is_flag=True,
    default=False,
    help="Do not save metrics, otherwise metrics are saved by default",
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
@click.option(
    "--lite",
    "model_fidelity",
    flag_value="lite",
    default="lite",
    help="Use lite MediaPipe model (default).",
)
@click.option(
    "--heavy",
    "model_fidelity",
    flag_value="heavy",
    help="heavy model.",
)
@click.option(
    "--angular-kinematics",
    "angular_kinematics_joints",
    type=click.Choice(
        ["left_knee", "right_knee", "left_elbow", "right_elbow"],
        case_sensitive=False,
    ),
    required=False,
    default=None,
    multiple=True,
    help="Which joint angular kinematics to plot.",
)
@click.option(
    "--linear-kinematics",
    "linear_kinematics_joints",
    type=click.Choice(
        [lm.name.lower() for lm in PoseLandmark], case_sensitive=False
    ),
    required=False,
    default=None,
    multiple=True,
    help="Which joint angular kinematics to plot.",
)
def main(video_path: Path,
    output_dir: Path,
    no_metrics: bool,
    debug_logging: bool,
    scale_factor: float,
    model_fidelity,
    angular_kinematics_joints,
    linear_kinematics_joints):
    """CLI wrapper: python main.py file.mp4"""
    process_video(
      video_path,
      output_dir,
      no_metrics,
      debug_logging
      scale_factor,
      model_fidelity,
      angular_kinematics_joints,
      linear_kinematics_joints
    )


if __name__ == "__main__":
    main()
