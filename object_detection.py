import argparse
import time
from typing import List, Any, Union
import imutils
import cv2
import numpy as np

# object detection demo
from utils.drawing import draw_crosshair, draw_dot, draw_annotations
from utils.utils import calculate_midpoint, VideoStream, FPS, get_frame_height_width

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file"
    )
    ap.add_argument(
        "-m", "--model", required=True, help="path to Caffe pre-trained model"
    )
    ap.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.2,
        help="minimum probability to filter weak detections",
    )

    arguments = vars(ap.parse_args())

    # do not change order
    AVAILABLE_CLASSES: List[Union[str, Any]] = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]

    DISPLAY_CLASSES: List[Union[str, Any]] = [
        "person",
    ]

    colours = np.random.uniform(0, 255, size=(len(AVAILABLE_CLASSES), 3))

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(arguments["prototxt"], arguments["model"])
    # initialize the video stream, allow the cammera sensor to warmup,
    # and initialize the FPS counter
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    fps = FPS().start()

    # loop over the frames from the video stream
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=1600)
        # grab the frame dimensions and convert it to a blob
        (height, width) = get_frame_height_width(frame)

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
        )
        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]
            # filter out weak detections:
            if confidence > arguments["confidence"]:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])

                if AVAILABLE_CLASSES[idx] not in DISPLAY_CLASSES:
                    continue

                box = detections[0, 0, i, 3:7] * np.array(
                    [width, height, width, height]
                )
                (startX, startY, endX, endY) = box.astype("int")

                midpoint = calculate_midpoint(startX, startY, endX, endY)
                print(f"x: {midpoint[0]}, y:{midpoint[0]}")

                draw_crosshair(frame, midpoint)

                draw_dot(frame, midpoint)

                draw_annotations(
                    frame=frame,
                    copy=[
                        f"Detected: {AVAILABLE_CLASSES[idx]}",
                        f"Confidence: {round(confidence * 100, 2)} %",
                        f"Coords: x:{str(midpoint[0])}, y:{str(midpoint[1])}",
                        f"Average FPS: 24.2",
                    ]
                )

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        # update the FPS counter
        fps.update()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()
    vs.stop()
