import cv2
import numpy as np
import gdown
import os
import re

from pathlib import Path
from pipeline.pipeline import (
    StageBase,
    VideoConfiguration,
    VideoData,
    Frame,
)

class VideoLoader(StageBase[VideoConfiguration, VideoData]):
    def execute(self, input: VideoConfiguration) -> VideoData:
        self.logger.info("Starting VideoLoader stage")
        
        # Resolve path & download Google Drive files, if needed
        resolved_path = _resolve_input_path(str(input.path))
        input.path = Path(resolved_path)
        
        # Ensure the name arg is set
        if input.name is None:
            # Use the input filename without extension
            input.name = input.path.stem
        
        cap = cv2.VideoCapture(str(input.path))
        fps = cap.get(cv2.CAP_PROP_FPS)

        frames = []
        msg = "Loading frames from video"
        if input.scale_factor is not None:
            msg += f", downscaling by {input.scale_factor}"
        self.logger.info(msg)

        while cap.isOpened():
            successful, frame = cap.read()

            if not successful:
                self.logger.info(
                    "Can't read frame, breaking out of video read"
                )
                break
            if input.scale_factor is not None:
                frame = cv2.resize(
                    frame,
                    dsize=None,
                    fx=input.scale_factor,
                    fy=input.scale_factor,
                )
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(
                Frame(frame, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
            )
        cap.release()

        self.logger.info(
            f"Loaded {len(frames)} frames from {str(input.path)}"
        )
        self.logger.info("Finished VideoLoader stage")

        return VideoData(frames=frames, fps=fps, config=input)


def _resolve_input_path(path: str) -> str:
    # Local file
    if os.path.exists(path):
        return path

    # Try to extract a Google Drive file ID
    try:
        file_id = extract_drive_file_id(path)
    except ValueError:
        raise FileNotFoundError(f"Path '{path}' not found and not a valid GDrive ID/URL.")

    # Download to media/realspeed
    download_dir = "media/realspeed"
    os.makedirs(download_dir, exist_ok=True)
    output_path = os.path.join(download_dir, f"{file_id}.mp4")

    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {file_id} from Google Drive...")
        gdown.download(url, output_path, quiet=False)

    return output_path


def extract_drive_file_id(path: str) -> str:
    """
    Extract the Google Drive file ID from a URL or return the ID itself if passed directly.

    Supports formats like:
        - https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
        - https://drive.google.com/open?id=<FILE_ID>
        - https://drive.google.com/uc?id=<FILE_ID>
        - Just the FILE_ID itself
    """

    # Normalize Windows backslashes
    normalized = path.replace("\\", "/")

    # Regex patterns for common Google Drive URL types
    patterns = [
        r"/d/([a-zA-Z0-9_-]{28,})",     # file/d/<ID>/view
        r"[?&]id=([a-zA-Z0-9_-]{28,})", # ?id=<ID> or &id=<ID>
        r"https://drive\.google\.com/uc\?export=download&id=([a-zA-Z0-9_-]{28,})"
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return match.group(1)

    # If the path looks like a valid file ID (28–33 chars, alphanumeric + _ -)
    if re.fullmatch(r"[a-zA-Z0-9_-]{28,33}", path):
        return path

    raise ValueError(f"Cannot extract Google Drive file ID from '{path}'")
