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
    WorldLandmarkLinearKinematicVariables,
)
from utils.joints import JOINTS, Joint
from utils.videoProcessingFunctions import draw_landmarks_on_image
from typing import Tuple, List
from pathlib import Path
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarkerResult,
)


class FuseVideoAndBoxingMetrics(
    StageBase[Tuple[VideoData, BoxingPunchMetrics, List[PoseLandmarkerResult]], FuncAnimation]
):
    def PlotJointLinearKinematics(
        self,
        input: Tuple[
            VideoData,
            WorldLandmarkLinearKinematicVariables,
            List[PoseLandmarkerResult],
        ],
        joint: PoseLandmark,
    ):
        video_data, kinematics, landmarkers = input
        pos = kinematics.position[:, joint]
        vel = kinematics.velocity[:, joint]
        accel = kinematics.acceleration[:, joint]
        fig = plt.figure(figsize=(14, 6), layout="tight")
        gs = gridspec.GridSpec(3, 2, figure=fig, width_ratios=[1, 3])
        ax_video = fig.add_subplot(gs[:, 0])
        ax_pos = fig.add_subplot(gs[0, 1])
        ax_vel = fig.add_subplot(gs[1, 1], sharex=ax_pos)
        ax_accel = fig.add_subplot(gs[2, 1], sharex=ax_vel)
        kin_axs = [ax_pos, ax_vel, ax_accel]
        joint_name = joint.name
        ax_pos.set(
            title=f"{joint_name} pos",
        )
        ax_vel.set(
            title=f"{joint_name} vel",
        )
        ax_accel.set(
            title=f"{joint_name} accel",
            xlabel="frame idx",
        )
        [ax.grid(True) for ax  in kin_axs]

        num_frames = len(video_data.frames)
        ax_pos.plot(range(num_frames), pos[:, 0], color='r', label='x')
        ax_pos.plot(range(num_frames), pos[:, 1], color='g', label='y')
        ax_pos.plot(range(num_frames), pos[:, 2], color='b', label='z')

        ax_vel.plot(range(num_frames), vel[:, 0], color='r', label='x')
        ax_vel.plot(range(num_frames), vel[:, 1], color='g', label='y')
        ax_vel.plot(range(num_frames), vel[:, 2], color='b', label='z')

        ax_accel.plot(range(num_frames), accel[:, 0], color='r', label='x')
        ax_accel.plot(range(num_frames), accel[:, 1], color='g', label='y')
        ax_accel.plot(range(num_frames), accel[:, 2], color='b', label='z')
        ax_pos.legend()
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
            interval=int(1e3 / video_data.fps),
            blit=True,
        )
        return anim

    def PlotJointAngularKinematics(
        self,
        input: Tuple[
            VideoData,
            JointAngularKinematicVariables,
            List[PoseLandmarkerResult],
        ],
        joint: PoseLandmark,
    ):
        video_data, kinematics, landmarkers = input
        pos = kinematics.joint_3d_angular_kinematics.position
        vel = kinematics.joint_3d_angular_kinematics.velocity
        accel = kinematics.joint_3d_angular_kinematics.acceleration

        fig = plt.figure(figsize=(14, 6), layout="tight")
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

        cursor_line_pos = ax_pos.axvline(0, color="k", linestyle="--")
        cursor_line_vel = ax_vel.axvline(0, color="k", linestyle="--")
        cursor_line_accel = ax_accel.axvline(
            0, color="k", linestyle="--"
        )
        # Display first video frame
        frame_rgb = cv2.cvtColor(
            video_data.frames[0].frame, cv2.COLOR_BGR2RGB
        )

        joint_landmarks = JOINTS[joint].get_landmarks()
        joint_connections = [(0, 1), (1, 2)]
        annotated_frame = draw_landmarks_on_image(
            frame_rgb,
            landmarkers[0],
            joint_landmarks,
            joint_connections,
        )
        im = ax_video.imshow(annotated_frame)
        ax_video.axis("off")

        def update(frame_idx):

            frame_rgb = cv2.cvtColor(
                video_data.frames[frame_idx].frame, cv2.COLOR_BGR2RGB
            )
            annotated_frame = draw_landmarks_on_image(
                frame_rgb,
                landmarkers[frame_idx],
                joint_landmarks,
                joint_connections,
            )
            im.set_data(annotated_frame)
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
            interval=int(1e3 / video_data.fps),
            blit=True,
        )
        return anim

    def execute(
        self,
        input: Tuple[VideoData, BoxingPunchMetrics,  List[PoseLandmarkerResult]],
    ) -> FuncAnimation:

        video_data, boxing_metrics, landmarkers = input

        # Figure with two rows: top for video, bottom for metrics
        fig = plt.figure(figsize=(14, 6))
        gs = gridspec.GridSpec(
            2, 3, figure=fig, width_ratios=[1, 3, 2]
        )

        ax_video = fig.add_subplot(gs[:, 0])
        ax_punch = fig.add_subplot(gs[0, 1])
        ax_rotation = fig.add_subplot(gs[1, 1])
        ax_com = fig.add_subplot(gs[:2, 2])

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
            color="orange",
            label="Left Wrist",
        )
        ax_punch.plot(
            range(num_frames),
            right_vel,
            color="purple",
            label="Right Wrist",
        )
        ax_punch.legend(loc="upper right")

        cursor_line_punch_metrics = ax_punch.axvline(
            0, color="k", linestyle="--"
        )

        # Plot rotation metrics
        cursor_line_rotation_metrics = None
        if boxing_metrics.hip_rotation_velocity_magnitude is not None:
            hip_vel = boxing_metrics.hip_rotation_velocity_magnitude
            ax_rotation.plot(
                range(num_frames),
                hip_vel,
                color="blue",
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
                color="red",
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

        ax_com.set_xlabel("X")
        ax_com.set_ylabel("Z")
        ax_com.set_title("Center of Mass")
        ax_com.grid(True)
        ax_com.axis("equal")
        ax_com.plot(
            boxing_metrics.center_of_mass[:, 0],
            boxing_metrics.center_of_mass[:, 2],
            color="green",
            marker='o',
            linestyle='None',
            alpha=0.2,
        )
        (com_marker,) = ax_com.plot(
            [], [], color='green', marker='*', animated=True, ms=20, mec="black", label='COM', linestyle='None'
        )
        idx1, idx2 = 0, 2
        def get_positions(ii, key):
            pos = getattr(boxing_metrics, key)
            return [pos[ii, idx1, 0], pos[ii, idx1, 1]], [pos[ii, idx2, 0], pos[ii, idx2, 1]]
        x_s, z_s = get_positions(0, 'shoulder_position')
        (shoulder_line, ) = ax_com.plot(x_s, z_s, label="shoulder", color='red')
        x_h, z_h = get_positions(0, 'hip_position')
        (hip_line, ) = ax_com.plot(x_h, z_h , label="hip", color='blue')
        x_f, z_f = get_positions(0, 'heel_position')
        (heel_line, ) = ax_com.plot(x_f, z_f, label="heel", color='black')
        ax_com.legend()
        xmin = min(np.min(boxing_metrics.shoulder_position[:, idx1, :]), np.min(boxing_metrics.hip_position[:, idx1, :]))
        xmax = max(np.max(boxing_metrics.shoulder_position[:, idx1, :]), np.max(boxing_metrics.hip_position[:, idx1, :]))
        zmin= min(np.min(boxing_metrics.shoulder_position[:, idx2, :]), np.min(boxing_metrics.hip_position[:, idx2, :]))
        zmax= max(np.max(boxing_metrics.shoulder_position[:, idx2, :]), np.max(boxing_metrics.hip_position[:, idx2, :]))
        ax_com.set(xlim=(xmin, xmax), ylim=(zmin, zmax))
        # Display first video frame
        frame_rgb = cv2.cvtColor(
            video_data.frames[0].frame, cv2.COLOR_BGR2RGB
        )

        joint_connections = [(0, 1)]
        annotated_frame = draw_landmarks_on_image(
            frame_rgb,
            landmarkers[0],
            [PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER],
            joint_connections,
        )
        annotated_frame = draw_landmarks_on_image(
            annotated_frame,
            landmarkers[0],
            [PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP],
            joint_connections,
        )
        annotated_frame = draw_landmarks_on_image(
            annotated_frame,
            landmarkers[0],
            [PoseLandmark.LEFT_HEEL, PoseLandmark.RIGHT_HEEL],
            joint_connections,
        )

        im = ax_video.imshow(annotated_frame)
        ax_video.axis("off")

        def update(frame_idx):
            # Update video frame
            frame_rgb = cv2.cvtColor(
                video_data.frames[frame_idx].frame, cv2.COLOR_BGR2RGB
            )
            joint_connections = [(0, 1)]
            annotated_frame = draw_landmarks_on_image(
                frame_rgb,
                landmarkers[frame_idx],
                [PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER],
                joint_connections,
            )
            annotated_frame = draw_landmarks_on_image(
                annotated_frame,
                landmarkers[frame_idx],
                [PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP],
                joint_connections,
            )
            annotated_frame = draw_landmarks_on_image(
                annotated_frame,
                landmarkers[frame_idx],
                [PoseLandmark.LEFT_HEEL, PoseLandmark.RIGHT_HEEL],
                joint_connections,
            )            
            im.set_data(annotated_frame)
            ax_video.set_title(f"Frame {frame_idx+1}/{num_frames}")
            com_marker.set_data(
                [boxing_metrics.center_of_mass[frame_idx, 0]],
                [boxing_metrics.center_of_mass[frame_idx, 2]],
            )
            cursor_line_punch_metrics.set_xdata([frame_idx])
            cursor_lines = [cursor_line_punch_metrics]
            if cursor_line_rotation_metrics:
                cursor_line_rotation_metrics.set_xdata([frame_idx])
                cursor_lines.append(cursor_line_rotation_metrics)
            x_h, z_h = get_positions(frame_idx, 'hip_position')
            hip_line.set_data(x_h, z_h)
            x_s, z_s = get_positions(frame_idx, 'shoulder_position')
            shoulder_line.set_data(x_s, z_s)
            x_f, z_f = get_positions(frame_idx, 'heel_position')
            heel_line.set_data(x_f, z_f)
            return cursor_lines + [im, com_marker, hip_line, shoulder_line, heel_line]

        anim = FuncAnimation(
            fig,
            update,
            frames=num_frames,
            interval=int(1e3 / video_data.fps),
            blit=True,
        )
        return anim
