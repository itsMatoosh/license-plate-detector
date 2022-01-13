import cv2
import os
import pandas as pd
import Localization
import Recognize

"""
    In this file, you will define your own CaptureFrame_Process funtion. In this function,
    you need three arguments: file_path(str type, the video file), sample_frequency(second), save_path(final results saving path).
    To do:
        1. Capture the frames for the whole video by your sample_frequency, record the frame number and timestamp(seconds).
        2. Localize and recognize the plates in the frame.(Hints: need to use 'Localization.plate_detection' and 'Recognize.segmetn_and_recognize' functions)
        3. If recognizing any plates, save them into a .csv file.(Hints: may need to use 'pandas' package)
    Inputs:(three)
        1. file_path: video path
        2. sample_frequency: second
        3. save_path: final .csv file path
    Output: None
"""


def CaptureFrame_Process(file_path, sample_frequency, save_path):
    # read video file
    vid = cv2.VideoCapture(file_path)

    # sample frames from the video and localize them
    fps = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_period = int((1 / sample_frequency) * fps)
    i = 0
    localized_plates = []
    while vid.isOpened():
        ret, frame = vid.read()
        if ret and i % sample_period == 0:
            localized_plates.append(Localization.plate_detection(frame))
        else:
            break
    vid.release()

    # recognize localized plates
    recognized_plates = Recognize.segment_and_recognize(localized_plates)

    # todo: save recognized plates into csv

    pass
