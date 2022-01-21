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
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    sample_period = int((1 / sample_frequency) * fps)
    print('Localizing license plate in video: ' + str(file_path) + ', Sampling every ' + str(sample_period) + ' frames')
    frame_num = 0
    i = 0
    localized_plates = []
    plate_metadata = []
    while True:
        ret, frame = vid.read()
        if ret:
            if (frame_num % sample_period) == 0:
                # append plate images
                plates = Localization.plate_detection(frame)
                for plate in plates:
                    localized_plates.append(plate)
                    plate_metadata.append([i, frame_num, ])
                    i += 1
            else:
                localized_plates.append([])
            frame_num += 1
        else:
            break
    vid.release()
    cv2.destroyAllWindows()
    print('Plates localized')

    # recognize localized plates
    print('Recognizing ' + str(len(localized_plates)) + ' frames with plates...')
    recognized_plates = Recognize.segment_and_recognize(localized_plates)

    # save recognized plates into csv
    for i in range(len(recognized_plates)):
        metadata = plate_metadata[i]
        characters = recognized_plates[i]


    pass
