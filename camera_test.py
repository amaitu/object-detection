from __future__ import print_function

import argparse
import datetime
import time

import cv2
import imutils

from utils.utils import VideoStream

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-p",
        "--picamera",
        type=int,
        default=-1,
        help="whether or not the Raspberry Pi camera should be used",
    )
    args = vars(ap.parse_args())
    vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        # draw the timestamp on the frame
        timestamp = datetime.datetime.now()
        ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
        cv2.putText(
            frame,
            ts,
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
        )
        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    vs.stop()
