import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

from mediapipe.python.solutions.pose import PoseLandmark

from matplotlib.animation import FuncAnimation
from pipeline.pipeline import (
    StageBase,
    VideoData,
    BoxingPunchMetrics,
    JointAngularKinematicVariables,
)
from utils.joints import JOINTS, Joint
from typing import Tuple
from pathlib import Path


class FuseVideoAndBoxingMetrics(
    StageBase[
        Tuple[VideoData, BoxingPunchMetrics],
        str,
    ]
):
    def PlotJointAngularKinematics(
        self, input: Tuple[VideoData, JointAngularKinematicVariables], joint: PoseLandmark
    ):
        video_data, kinematics = input
        pos = kinematics.joint_3d_angular_kinematics.position
        vel = kinematics.joint_3d_angular_kinematics.velocity
        accel = kinematics.joint_3d_angular_kinematics.acceleration

        fig = plt.figure(figsize=(14, 6), layout='tight')
        gs = gridspec.GridSpec(3, 2, figure=fig, width_ratios=[1, 3])
        ax_video = fig.add_subplot(gs[:, 0])
        ax_pos = fig.add_subplot(gs[0, 1])
        ax_vel = fig.add_subplot(gs[1, 1], sharex=ax_pos)
        ax_accel = fig.add_subplot(gs[2, 1], sharex=ax_vel)
        joint_name = joint.name
        ax_pos.set(
            title=f"{joint_name} angular pos",
            xlabel="frame idx",
            ylabel="angle",
        )
        ax_vel.set(
            title=f"{joint_name} angular vel",
            xlabel="frame idx",
            ylabel="angle/s",
        )
        ax_accel.set(
            title=f"{joint_name} angular accel",
            xlabel="frame idx",
            ylabel="angle/s^2",
        )
        ax_pos.grid(True)
        ax_vel.grid(True)
        ax_accel.grid(True)

        num_frames = len(video_data.frames)

        ax_pos.plot(
            range(num_frames),
            pos[:, JOINTS[joint].index],
            color="r",
        )
        ax_vel.plot(
            range(num_frames),
            vel[:, JOINTS[joint].index],
            color="g",
        )
        ax_accel.plot(
            range(num_frames),
            accel[:, JOINTS[joint].index],
            color="b",
        )

        punch_frames_rear_hook = [90, 159, 233, 350, 422, 492, 617, 675, 742]
        punch_frames_uppercut = [28, 100, 172, 309, 371, 434, 630, 687]
        window=3
        key_frames = punch_frames_rear_hook
        if 'uppercut' in video_data.config.name:
            key_frames = punch_frames_uppercut
        for f in key_frames:
            for ax in [ax_pos, ax_vel, ax_accel]:
                ax.axvspan(f - window, f + window, color="y", alpha=0.2)      

        cursor_line_pos = ax_pos.axvline(0, color="k", linestyle="--")
        cursor_line_vel = ax_vel.axvline(0, color="k", linestyle="--")
        cursor_line_accel = ax_accel.axvline(
            0, color="k", linestyle="--"
        )
        # Display first video frame
        frame_rgb = cv2.cvtColor(
            video_data.frames[0].frame, cv2.COLOR_BGR2RGB
        )
        im = ax_video.imshow(frame_rgb)
        ax_video.axis("off")

        def update(frame_idx):

            frame_rgb = cv2.cvtColor(
                video_data.frames[frame_idx].frame, cv2.COLOR_BGR2RGB
            )
            im.set_data(frame_rgb)
            ax_video.set_title(f"Frame {frame_idx+1}/{num_frames}")
            cursor_line_pos.set_xdata([frame_idx])
            cursor_line_vel.set_xdata([frame_idx])
            cursor_line_accel.set_xdata([frame_idx])
            cursor_lines = [
                cursor_line_pos,
                cursor_line_vel,
                cursor_line_accel,
            ]
            return cursor_lines + [im]

        anim = FuncAnimation(
            fig,
            update,
            frames=num_frames,
            interval=1000 / 15,
            blit=True,
        )
        
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_filename = (
            f"{video_data.config.path.stem}_{joint.name}_angular_kinematics.mp4"
        )
        output_path = output_dir / output_filename
        plt.show()
        anim.save(output_path, writer="ffmpeg", fps=15)
        return output_path

    def execute(
        self,
        input: Tuple[VideoData, BoxingPunchMetrics],
    ) -> str:

        video_data, boxing_metrics = input

        # Figure with two rows: top for video, bottom for metrics
        fig = plt.figure(figsize=(14, 6))
        gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1, 3])

        ax_video = fig.add_subplot(gs[:, 0])

        ax_punch = fig.add_subplot(gs[0, 1])
        ax_rotation = fig.add_subplot(gs[1, 1])

        ax_rotation.sharex(ax_punch)

        num_frames = len(video_data.frames)

        # Plot wrist velocity metrics
        left_vel = (
            boxing_metrics.left_wrist_punching_velocity_magnitude
        )
        right_vel = (
            boxing_metrics.right_wrist_punching_velocity_magnitude
        )

        ax_punch.set(
            xlabel="Frame Index",
            ylabel="Velocity Magnitude (cm/s)",
        )
        ax_punch.grid(True)

        ax_punch.plot(
            range(num_frames),
            left_vel,
            color="blue",
            label="Left Wrist",
        )
        ax_punch.plot(
            range(num_frames),
            right_vel,
            color="red",
            label="Right Wrist",
        )
        ax_punch.legend(loc="upper right")

        cursor_line_punch_metrics = ax_punch.axvline(
            0, color="k", linestyle="--"
        )

        # Plot rotation metrics
        cursor_line_rotation_metrics = None
        if (
            boxing_metrics.hip_rotation_velocity_magnitude is not None
            or boxing_metrics.shoulder_rotation_velocity_magnitude
            is not None
        ):
            if (
                boxing_metrics.hip_rotation_velocity_magnitude
                is not None
            ):
                hip_vel = (
                    boxing_metrics.hip_rotation_velocity_magnitude
                )
                ax_rotation.plot(
                    range(num_frames),
                    hip_vel,
                    color="purple",
                    label="hip",
                )

            if (
                boxing_metrics.shoulder_rotation_velocity_magnitude
                is not None
            ):
                shoulder_vel = (
                    boxing_metrics.shoulder_rotation_velocity_magnitude
                )
                ax_rotation.plot(
                    range(num_frames),
                    shoulder_vel,
                    color="orange",
                    label="shoulder",
                )
            ax_rotation.set(
                xlabel="Frame Index",
                ylabel="Rotation magnitude",
            )
            ax_rotation.grid(True)

            ax_rotation.legend(loc="upper right")
            cursor_line_rotation_metrics = ax_rotation.axvline(
                0, color="k", linestyle="--"
            )

        # Display first video frame
        frame_rgb = cv2.cvtColor(
            video_data.frames[0].frame, cv2.COLOR_BGR2RGB
        )
        im = ax_video.imshow(frame_rgb)
        ax_video.axis("off")

        def update(frame_idx):
            # Update video frame
            frame_rgb = cv2.cvtColor(
                video_data.frames[frame_idx].frame, cv2.COLOR_BGR2RGB
            )
            im.set_data(frame_rgb)
            ax_video.set_title(f"Frame {frame_idx+1}/{num_frames}")
            cursor_line_punch_metrics.set_xdata([frame_idx])
            cursor_lines = [cursor_line_punch_metrics]
            if cursor_line_rotation_metrics:
                cursor_line_rotation_metrics.set_xdata([frame_idx])
                cursor_lines.append(cursor_line_rotation_metrics)

            return cursor_lines + [im]

        anim = FuncAnimation(
            fig,
            update,
            frames=num_frames,
            interval=1000 / 15,  # ~15 FPS
            blit=True,
        )

        # Save the animation and its output path
        output_path = define_output_path(video_data.config.path)
        anim.save(output_path, writer="ffmpeg", fps=15)
        plt.show()
        return output_path


def define_output_path(video_path: Path) -> str:

    # --- Build output path ---
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_filename = f"{video_path.stem}_with_metrics.mp4"
    output_path = output_dir / output_filename

    return str(output_path)
