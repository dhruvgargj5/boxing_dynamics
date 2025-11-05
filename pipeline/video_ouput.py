import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2


from matplotlib.animation import FuncAnimation
from pipeline.pipeline import StageBase, VideoData, BoxingPunchMetrics
from typing import Tuple
from pathlib import Path


class FuseVideoAndBoxingMetrics(
    StageBase[
        Tuple[VideoData, BoxingPunchMetrics],
        str,
    ]
):
    def execute(
        self,
        input: Tuple[VideoData, BoxingPunchMetrics],
    ) -> str:

        video_data, boxing_metrics = input

        # Figure with two rows: top for video, bottom for metrics
        fig = plt.figure(figsize=(14, 6))
        gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1,3])

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
            or boxing_metrics.shoulder_rotation_velocity_magnitude is not None
        ):
            if boxing_metrics.hip_rotation_velocity_magnitude is not None:
                hip_vel = (
                    boxing_metrics.hip_rotation_velocity_magnitude
                )
                ax_rotation.plot(
                    range(num_frames),
                    hip_vel,
                    color="purple",
                    label="hip",
                )

            if boxing_metrics.shoulder_rotation_velocity_magnitude is not None:
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
            cursor_line_rotation_metrics = (
                ax_rotation.axvline(
                    0, color="k", linestyle="--"
                )
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
