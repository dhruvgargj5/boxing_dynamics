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
        self, input: Tuple[VideoData, BoxingPunchMetrics], video_path: str
    ) -> str:

        video_data, boxing_metrics = input

        # Figure with two rows: top for video, bottom for metrics
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax_video = fig.add_subplot(gs[0])
        ax_plot = fig.add_subplot(gs[1])

        num_frames = len(video_data.frames)

        # Plot metrics
        left_vel = boxing_metrics.left_wrist_punching_velocity_magnitude
        right_vel = boxing_metrics.right_wrist_punching_velocity_magnitude

        ax_plot.set(
            xlim=(0, num_frames),
            ylim=(0, max(np.max(left_vel), np.max(right_vel)) * 1.1),
            xlabel="Frame Index",
            ylabel="Velocity Magnitude (cm/s)",
        )
        ax_plot.grid(True)

        ax_plot.plot(range(num_frames), left_vel, color='blue', label="Left Wrist")
        ax_plot.plot(range(num_frames), right_vel, color='red', label="Right Wrist")
        ax_plot.legend(loc="upper right")

        cursor_line = ax_plot.axvline(0, color="k", linestyle="--")

        # Display first video frame
        frame_rgb = cv2.cvtColor(video_data.frames[0].frame, cv2.COLOR_BGR2RGB)
        im = ax_video.imshow(frame_rgb)
        ax_video.axis("off")

        def update(frame_idx):
            # Update video frame
            frame_rgb = cv2.cvtColor(video_data.frames[frame_idx].frame, cv2.COLOR_BGR2RGB)
            im.set_data(frame_rgb)
            ax_video.set_title(f"Frame {frame_idx+1}/{num_frames}")

            # Move cursor line
            cursor_line.set_xdata([frame_idx])

            return im, cursor_line

        anim = FuncAnimation(
            fig,
            update,
            frames=num_frames,
            interval=1000 / 15,  # ~15 FPS
            blit=True,
        )

        # Save the animation and its output path
        output_path = define_output_path(video_path)
        anim.save(output_path, writer="ffmpeg", fps=15)
        plt.close(fig)

        return output_path

def define_output_path(video_path: str) -> str:

    # --- Build output path ---
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_filename = f"{video_path.stem}_with_metrics.mp4"
    output_path = output_dir / output_filename

    return str(output_path)