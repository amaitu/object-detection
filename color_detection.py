import argparse
import time

import cv2
import numpy
import pkg_resources

from utils.utils import VideoStream, FPS, calculate_midpoint, OutputLogger


def convert_frame_to_colour_model(image_frame):
    """
    Convert the imageFrame in
    BGR(RGB color space) to
    HSV(hue-saturation-value)
    color space
    https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0aa4a7f0ecf2e94150699e48c79139ee12
    """
    return cv2.cvtColor(image_frame, cv2.COLOR_BGR2HSV)


def create_mask(frame):
    start, end = COLOUR_THRESHOLDS[args["colour"]]
    return cv2.inRange(frame, start, end)


if __name__ == "__main__":
    COLOUR_THRESHOLDS = {
        "blue": ((94, 80, 2), (120, 255, 255)),
        "yellow": ((15, 0, 0), (36, 255, 255)),
        "red": ((136, 87, 111), (180, 255, 255)),
        "green": ((25, 52, 72), (102, 255, 255)),
    }

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-p", "--colour", required=True, help="colour, one of: red,blue,green,yellow",
    )

    ap.add_argument(
        "-fr", "--framerate", default=32, help="FPS, integer",
    )

    args = vars(ap.parse_args())

    if args["colour"] not in COLOUR_THRESHOLDS.keys():
        raise RuntimeError("Invalid colour.")

    print(f"[INFO] seeking: {args['colour']}")

    video_stream = VideoStream(src=0, framerate=args["framerate"]).start()
    time.sleep(2.0)
    fps = FPS().start()

    output_logger = OutputLogger()

    while True:
        imageFrame = video_stream.read()
        hsv_frame = convert_frame_to_colour_model(imageFrame)

        cv2.putText(
            imageFrame,
            f"MakeSense {pkg_resources.get_distribution('object-detection').version}",
            (10, hsv_frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
        )

        mask = create_mask(hsv_frame)

        # Morphological Transform, Dilation
        # for each color and bitwise_and operator
        # between imageFrame and mask determines
        # to detect only that particular color
        kernel = numpy.ones((5, 5), "uint8")

        mask = cv2.dilate(mask, kernel)
        cv2.bitwise_and(imageFrame, imageFrame, mask=mask)

        # Creating contours to track color
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 300:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            output = output_logger.set_output(calculate_midpoint(x, y, x + w, y + h))

            coordinates_to_draw = (
                calculate_midpoint(x, y, x + w, y + h)
                if output
                else output_logger.current_output
            )

            cv2.circle(
                imageFrame, coordinates_to_draw, radius=0, color=(0, 0, 0), thickness=4,
            )
            cv2.putText(
                imageFrame,
                str(args["colour"]),
                (coordinates_to_draw[0] + 15, coordinates_to_draw[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color=(0, 0, 0),
            )

        cv2.imshow("colours", imageFrame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
