#!/usr/bin/env python3

import click
import logging
from pathlib import Path

from pipeline.pipeline import (
    BoxingDynamicsPipeline,
    StageBase,
    VideoConfiguration,
    VideoData,
)
from pipeline.video_loader import VideoLoader
from pipeline.landmarking import ExtractHumanPoseLandmarks


@click.command()
@click.option(
    "--debug-logging",
    is_flag=True,
    help="Enable DEBUG logging",
    default=False,
)
def main(debug_logging: bool):

    log_level = logging.DEBUG if debug_logging else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    logging.info("Starting BoxingDynamics pipeline")

    first_input = VideoConfiguration(
        name="Max's Cross Punch",
        path=Path("media/realspeed/cross.MP4"),
    )

    pipeline = BoxingDynamicsPipeline(
        [VideoLoader(), ExtractHumanPoseLandmarks()]
    )

    pipeline.run(first_input)

    logging.info("Finished BoxingDynamics pipeline")


if __name__ == "__main__":
    main()
