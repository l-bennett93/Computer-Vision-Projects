import argparse
import cv2

from vision_projects.utils import VideoIterator
from vision_projects.hand_landmark import HandDetector

parser = argparse.ArgumentParser(description='Detects the Euclidean distance between pinky and thumb and thresholds the video')
parser.add_argument('videoname', metavar='v', type=str, default=0)
parser.add_argument('shaka_threshold', metavar='t', type=int, default=300 ,help="distance of shaka before thresholding")

def distance_threshold(frame, point1_lm, point2_lm, threshold=300):
    """
    Thresholds the frame if the euclidean distance between the two
    points exceeds the threshold.

    Parameters
    ---------
    frame: np.ndarray
        The frame to be drawn upon

    point1_lm: tuple
        (x,y) tuple containing the coordinates of the first point

    point2_lm: tuple
        (x,y) tuple containing the coordinates of the second point

    threshold: int
        The distance required for the frame to be thresholded

    Returns
    -------
    frame: np.ndarray
        The frame thresholded if the distance is exceeded

    """
    if point1_lm and point2_lm:
        width_sqr = np.power(point1_lm[0] - point2_lm[0], 2)
        height_sqr =  np.power(point1_lm[1] - point2_lm[1], 2)
        l2_distance = np.sqrt(width_sqr + height_sqr)

        cv2.putText(
            frame,
            f"Distance: {l2_distance:.0f}",
            (int(frame.shape[1]*0.05),int(frame.shape[0]*0.95)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,0,255), 2
        )
        if l2_distance > threshold:
            _, frame = cv2.threshold(frame, 200, 255, cv2.THRESH_BINARY_INV)

    return frame

def main(iterator, model, shaka_threshold):
    for frame in frame_iterator:
        frame, landmarks = model(frame)

        thumb_lm = model.find_position(hand_number=0, landmark_idx=4)
        pinky_lm = model.find_position(hand_number=0, landmark_idx=20)

        # Threshold if the distance between the pinky and thumb is exceeded
        frame = distance_threshold(frame, thumb_lm, pinky_lm, threshold=shaka_threshold)

        cv2.imshow("frame", frame)

if __name__ == "__main__":
    #Parse args from command line
    args = parser.parse_args()
    video_name = args.video_name
    shaka_threshold = args.shaka_threshold

    #Set up the frame iterator and the model
    frame_iterator = VideoIterator(video_name)
    model = HandDetector()

    main(frame_iterator, model, shaka_threshold)
