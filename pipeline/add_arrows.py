"""
add_arrows.py

Draws one arrow on each foot (left and right ankle) for every frame in a video.

Input:
    Tuple[VideoData, List[PoseLandmarkerResult]]
Output:
    VideoData with arrows drawn on both feet
"""

import cv2
import numpy as np
from typing import Tuple, List
# from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from pipeline.pipeline import StageBase, AddArrowsStageInput, VideoData
from pathlib import Path


class AddArrowsToLegs(
    StageBase[AddArrowsStageInput, VideoData]
):
    """
    Draws simple arrows at the left and right ankle landmarks.
    """

    def __init__(
        self,
        color=(0, 255, 0),
        thickness: int = 3,
        arrow_length: int = 50,
    ):
        """
        Parameters
        ----------
        color : tuple
            BGR color for arrows.
        thickness : int
            Arrow line thickness.
        arrow_length : int
            Length of the foot arrow in pixels.
        """
        self.color = color
        self.thickness = thickness
        self.arrow_length = arrow_length

    def execute(self, input: AddArrowsStageInput ) -> VideoData:
        
        """        
        # MP_POSES= { "left_hip": 23, 
        #             "right_hip": 24,
        #             "left knee" : 25,
        #             "right_knee" : 26,
        #             "left_ankle": 27, 
        #             "right_ankle": 28,
        #             "left_heel": 29, 
        #             "right_heel" : 30,
        #             "left_foot_index" : 31, 
        #             "right_fot_index" : 32                 
        #             }
        
        """

        video_data = input.video_data 
        landmark_results = input.landmarkers
        save_video = input.save_video

        new_frames = []
        height, width, _ = video_data.frames[0].frame.shape

        for frame_idx, frame_data in enumerate(video_data.frames):

            # copy the video frame
            frame = frame_data.frame.copy()
            
            # get list of landmarks for this frame
            frame_landmarks = landmark_results[frame_idx].pose_landmarks[0]
            
            # Get coordinates for landmarks of interest in pixels
            landmarks_of_interest = {"left_hip" : 23, 
                                     "right_hip" : 24, 
                                     "left_knee": 25, 
                                     "right_knee" : 26}
            pts = self._extract_position_pts(landmarks_of_interest, frame_landmarks, width, height)

            # Draw one arrow per ankle (pointing slightly forward)
            for name, pt in pts.items():
                if pt is not None:
                    pt_start = pt
                    pt_end = (pt_start[0] + self.arrow_length, pt_start[1] + self.arrow_length // 2)
                    cv2.arrowedLine(frame, pt_start, pt_end, self.color, self.thickness, tipLength=0.3)

            new_frame_data = type(frame_data)(
                frame = frame, 
                timestamp_ms=frame_data.timestamp_ms
            )
            new_frames.append(new_frame_data)

        # overwrite old video_data 
        video_data = type(video_data)(
            frames = new_frames, 
            fps = video_data.fps,
            config=video_data.config)

        if save_video: 
            output_path = self.save_video(video_data)
            return video_data, output_path
        else:
            return video_data


    def _extract_position_pts(self, landmarks_of_interest:list, frame_landmarks, width: int, height: int ):
        """
        Extract the coordinates (in pixels) for landmarks of interest from pose landmarks.
        Expects pose_landmarks to be a list of NormalizedLandmark for a specific frame.
        """
        pts= {}
        
        for landmark_name, idx in landmarks_of_interest.items():
                if idx < len(frame_landmarks):
                    lm = frame_landmarks[idx]
                    x_px = int(lm.x * width)
                    y_px = int(lm.y * height)
                    pts[landmark_name] = (x_px, y_px)
        return pts
    
            
    def save_video(self, video_data):
        """Helper: save VideoData frames to MP4."""
        height, width, _ = video_data.frames[0].frame.shape
        fps = 15  # or video_data.config.fps if available

        video_path = video_data.config.path
        output_path = Path("output") / f"{video_path.stem}_with_arrows.mp4"
        output_path.parent.mkdir(exist_ok=True)

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        for frame_data in video_data.frames:
            writer.write(frame_data.frame)
        writer.release()
        
        return output_path