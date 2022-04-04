import logging
import os

from rich.logging import RichHandler


def setup_logger(exp_name, debug):
    from imp import reload

    reload(logging)

    CUDA_TAG = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    EXP_TAG = exp_name

    logger_config = dict(
        level=logging.DEBUG if debug else logging.INFO,
        format=f"{CUDA_TAG}:[{EXP_TAG}] %(message)s",
        handlers=[RichHandler()],
        datefmt="[%X]",
    )
    logging.basicConfig(**logger_config)