"""Localization module detects the location of license plates in an image."""
import string

import numpy as np
import cv2


def preprocess(image: np.ndarray):
    """Preprocesses an image before localization."""


def localize_plate(image: np.ndarray):
    """Localizes a license plate from an image."""


def localize(image: np.ndarray):
    """Performs localization on the given image"""


def localize_video(video_path: string):
    """Performs localization on every frame of a given video."""
    vid = cv2.VideoCapture(video_path)
    localizations = []
    while vid.isOpened():
        ret, frame = vid.read()
        if ret:
            localizations.append(localize(frame))
        else:
            break
    vid.release()
