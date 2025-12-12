from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import pose as mp_pose

import mediapipe as mp
import numpy as np
import cv2
from typing import List, Optional, Tuple


def get_keypoint_name_from_index(index):
    """Get the name of a keypoint given its index."""
    for name, idx in mp_pose.PoseLandmark.__members__.items():
        if idx.value == index:
            return name
    raise KeyError(f"Index {index} not found in PoseLandmark.")


def get_landmarker_results_from_video(
    video_path, options, start_time_ms=None, end_time_ms=None
) -> list:
    """Process a video file to extract pose landmarks using Mediapipe Pose Landmarker.
    Args:
        video_path: Path to the input video file.
        options: Configuration options for the Mediapipe Pose Landmarker.
        start_time_ms: Optional start time (in milliseconds) to begin processing the video.
        end_time_ms: Optional end time (in milliseconds) to stop processing the video.

    Returns:
        A list of dictionaries, each containing:
            - 'timestamp_ms': Timestamp of the frame in milliseconds.
            - 'original_frame': The downscaled video frame (as a NumPy array).
            - 'landmarker_results': The pose landmarker output for that frame.
    """

    # Open the video file for reading
    cap = cv2.VideoCapture(video_path)

    # Get the video frame rate (frames per second)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"{fps=}")

    # If a start time is provided, seek the video to that timestamp (in milliseconds)
    if start_time_ms:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time_ms)

    # Container for storing pose landmark detection results
    pose_landmarker_results = []

    # Create a Mediapipe pose landmarker using the provided configuration options
    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
        options
    )

    # Read video frames in a loop until the end
    while cap.isOpened():
        ret, frame = cap.read()  # Grab the next frame
        if not ret:
            # If no frame was read, break out of the loop
            print(f"Can't read frame. Skipping...")
            break

        # Current timestamp of the frame in milliseconds
        curr_frame_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        # If an end time is set and we've reached/passed it, stop processing
        if end_time_ms and curr_frame_timestamp_ms >= end_time_ms:
            break

        # Downscale the frame (reduce resolution by half for efficiency)
        downscaled_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Convert frame from OpenCV BGR format to Mediapipe SRGB format
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(downscaled_frame, cv2.COLOR_BGR2RGB),
        )

        # Run Mediapipe pose detection for this video frame at the given timestamp
        pose_landmarker_result = landmarker.detect_for_video(
            mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC))
        )

        # Store results (timestamp, frame, and landmarker output)
        pose_landmarker_results.append(
            {
                "timestamp_ms": curr_frame_timestamp_ms,
                "original_frame": downscaled_frame,
                "landmarker_results": pose_landmarker_result,
            }
        )

    # Release the video file handle
    cap.release()

    # Return the full list of detection results
    return pose_landmarker_results


def draw_landmarks_on_image(
    rgb_image,
    detection_result,
    landmarks_to_draw: Optional[List[mp_pose.PoseLandmark]] = None,
    landmark_connections: Optional[List[Tuple]] = None,
):
    if landmarks_to_draw is None:
        landmarks_to_draw = [lm for lm in mp_pose.PoseLandmark]
    pose_landmarks_list = detection_result.pose_landmarks
    pose_landmarks = pose_landmarks_list[0]
    annotated_image = np.copy(rgb_image)
    connections = (
        solutions.pose.POSE_CONNECTIONS
        if landmark_connections is None
        else landmark_connections
    )
    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend(
        [
            landmark_pb2.NormalizedLandmark(
                x=landmark.x, y=landmark.y, z=landmark.z
            )
            for idx, landmark in enumerate(pose_landmarks)
            if mp_pose.PoseLandmark(idx) in landmarks_to_draw
        ]
    )
    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        connections,
        solutions.drawing_styles.get_default_pose_landmarks_style(),
    )
    return annotated_image


def annotate_video(video_path, pose_landmarker_results):
    """Add media pose tracking to the video"""

    first_frame = pose_landmarker_results[0]["original_frame"]
    height, width, _ = first_frame.shape
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    out = cv2.VideoWriter(
        "media/annotated/hook.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 2, height),  # width doubled for side-by-side
    )

    for frame_data in pose_landmarker_results:
        original_frame = cv2.cvtColor(
            frame_data["original_frame"], cv2.COLOR_BGR2RGB
        )
        landmarks_frame = np.copy(original_frame)
        alpha = 0.5
        landmarks_frame = cv2.addWeighted(
            landmarks_frame,
            alpha,
            np.zeros_like(landmarks_frame),
            1 - alpha,
            0,
        )
        landmarks_frame = draw_landmarks_on_image(
            landmarks_frame, frame_data["landmarker_results"]
        )
        landmarkers = frame_data["landmarker_results"]

        side_by_side = np.concatenate(
            (original_frame, landmarks_frame), axis=1
        )

        side_by_side_bgr = cv2.cvtColor(
            side_by_side, cv2.COLOR_RGB2BGR
        )
        out.write(side_by_side_bgr)
    out.release()


def drawFBD(pose_landmarkers, frame_idx, original_frame_bgr):
    """draws a free body diagram on a given frame"""

    # Create a copy of the original frame to draw on
    original_frame_rgb = cv2.cvtColor(
        original_frame_bgr, cv2.COLOR_BGR2RGB
    )
    fbd_frame = original_frame_rgb.copy()
    img_height, img_width, _ = fbd_frame.shape

    # Extract 3D pose landmarks for this frame
    pose_landmarkers_4_frame = pose_landmarkers[frame_idx][
        "landmarker_results"
    ]

    left_shoulder_x, left_shoulder_y = (
        int(
            pose_landmarkers_4_frame.pose_landmarks[0][
                mp_pose.PoseLandmark.LEFT_SHOULDER
            ].x
            * img_width
        ),
        int(
            pose_landmarkers_4_frame.pose_landmarks[0][
                mp_pose.PoseLandmark.LEFT_SHOULDER
            ].y
            * img_height
        ),
    )

    right_shoulder_x, right_shoulder_y = (
        int(
            pose_landmarkers_4_frame.pose_landmarks[0][
                mp_pose.PoseLandmark.RIGHT_SHOULDER
            ].x
            * img_width
        ),
        int(
            pose_landmarkers_4_frame.pose_landmarks[0][
                mp_pose.PoseLandmark.RIGHT_SHOULDER
            ].y
            * img_height
        ),
    )

    left_hip_x, left_hip_y = (
        int(
            pose_landmarkers_4_frame.pose_landmarks[0][
                mp_pose.PoseLandmark.LEFT_HIP
            ].x
            * img_width
        ),
        int(
            pose_landmarkers_4_frame.pose_landmarks[0][
                mp_pose.PoseLandmark.LEFT_HIP
            ].y
            * img_height
        ),
    )

    right_hip_x, right_hip_y = (
        int(
            pose_landmarkers_4_frame.pose_landmarks[0][
                mp_pose.PoseLandmark.RIGHT_HIP
            ].x
            * img_width
        ),
        int(
            pose_landmarkers_4_frame.pose_landmarks[0][
                mp_pose.PoseLandmark.RIGHT_HIP
            ].y
            * img_height
        ),
    )

    # # Draw pose landmarks on the frame for reference
    # fbd_frame = draw_landmarks_on_image( fbd_frame, pose_landmarkers_4_frame)

    # # Draw a line between the left and right shoulders
    # cv2.line(fbd_frame, (left_shoulder_x, left_shoulder_y), (right_shoulder_x, right_shoulder_y), (255, 0, 0), 2)

    # # Draw a line between the left and right hips
    # cv2.line(fbd_frame, (left_hip_x, left_hip_y), (right_hip_x, right_hip_y), (255, 0, 0), 2)

    # # Draw a line between the left shoulder and left hip
    # cv2.line(fbd_frame, (left_shoulder_x, left_shoulder_y), (left_hip_x, left_hip_y), (255, 0, 0), 2)

    # # Draw a line between the right shoulder and right hip
    # cv2.line(fbd_frame, (right_shoulder_x, right_shoulder_y), (right_hip_x, right_hip_y), (255, 0, 0), 2)

    # determine the mid-hip point
    mid_hip_x = int((left_hip_x + right_hip_x) / 2)
    mid_hip_y = int((left_hip_y + right_hip_y) / 2)

    # determine the mid-shoulder point
    mid_shoulder_x = int((left_shoulder_x + right_shoulder_x) / 2)
    mid_shoulder_y = int((left_shoulder_y + right_shoulder_y) / 2)

    # draw COM as a red circle between mid-shoulder and mid-hip
    com_x = int((mid_shoulder_x + mid_hip_x) / 2)
    com_y = int((mid_shoulder_y + mid_hip_y) / 2)
    cv2.circle(fbd_frame, (com_x, com_y), 7, (255, 0, 0), -1)

    # draw a force arrow from COM downwards
    arrow_start = (com_x, com_y)
    arrow_end = (
        com_x,
        com_y + int(9.81 * img_height // 100),
    )  # length proportional to 9.81 m/s²
    cv2.arrowedLine(
        fbd_frame,
        arrow_start,
        arrow_end,
        (255, 0, 0),
        2,
        tipLength=0.3,
    )

    # draw ground reaction force arrow at the left foot based on which foot is down
    left_foot_y = int(
        pose_landmarkers_4_frame.pose_landmarks[0][
            mp_pose.PoseLandmark.LEFT_FOOT_INDEX
        ].y
        * img_height
    )
    right_foot_y = int(
        pose_landmarkers_4_frame.pose_landmarks[0][
            mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
        ].y
        * img_height
    )

    # left foot is down if right hip is rotated more than left hip or if left foot is lower than right foot
    left_foot_x, left_foot_y = (
        int(
            pose_landmarkers_4_frame.pose_landmarks[0][
                mp_pose.PoseLandmark.LEFT_FOOT_INDEX
            ].x
            * img_width
        ),
        int(
            pose_landmarkers_4_frame.pose_landmarks[0][
                mp_pose.PoseLandmark.LEFT_FOOT_INDEX
            ].y
            * img_height
        ),
    )

    right_foot_x, right_foot_y = (
        int(
            pose_landmarkers_4_frame.pose_landmarks[0][
                mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
            ].x
            * img_width
        ),
        int(
            pose_landmarkers_4_frame.pose_landmarks[0][
                mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
            ].y
            * img_height
        ),
    )

    if left_foot_y > right_foot_y or right_hip_x < left_hip_x:
        # left foot is down, draw GRF arrow at left foot
        grf_start = (left_foot_x, left_foot_y)
        grf_end = (
            left_foot_x,
            left_foot_y - int(9.81 * img_height // 100),
        )  # length proportional to 9.81 m/s²
        cv2.arrowedLine(
            fbd_frame,
            grf_start,
            grf_end,
            (0, 255, 0),
            2,
            tipLength=0.3,
        )

    else:
        # right foot is down, draw GRF arrow at right foot
        grf_start = (right_foot_x, right_foot_y)
        grf_end = (
            right_foot_x,
            right_foot_y - int(9.81 * img_height // 100),
        )  # length proportional to 9.81 m/s²
        cv2.arrowedLine(
            fbd_frame,
            grf_start,
            grf_end,
            (0, 255, 0),
            2,
            tipLength=0.3,
        )

    return fbd_frame
