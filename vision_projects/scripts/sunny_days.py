import cv2
import mediapipe as mp
import numpy as np

from vision_projects.utils import VideoIterator
from vision_projects.pose_estimation import PoseDetector

parser = argparse.ArgumentParser(description='Draws a pair of glasses on the face of the pose detection')
parser.add_argument('videoname', metavar='v', type=str, default=0)


def draw_glasses(frame, model):
    """
    Draws a pair of glasses on the face of the pose person

    Parameters
    ---------
    frame: np.ndarray
        The frame to be drawn upon

    model: PoseDetector
        The model to be used to create the facial land marks

    Returns
    -------
    frame: np.ndarray
        The frame with the glasses drawn on if pose is detected

    """
    if results.pose_landmarks:
        # eye coordinates
        left_eye_coords=model.find_position(landmark_idx=2)
        right_eye_coords=model.find_position(landmark_idx=5)

        # ear coordinates
        left_ear_coords=model.find_position(landmark_idx=7)
        right_ear_coords=model.find_position(landmark_idx=8)

        # distance between the ears used to scale glasses
        l22 = np.power(left_ear_coords[0] - right_ear_coords[0], 2) * np.power(left_ear_coords[1] - right_ear_coords[1], 2)
        l2_distance = np.sqrt(l22)

        #Manually selected, scales the glasses to some degree based on distance
        glasses_size = int(np.log(max(l2_distance,1.1))) * 3

        #Draw circles for eyes
        frame = cv2.circle(frame, left_eye_coords, glasses_size, (0, 0, 0), -1)
        frame = cv2.circle(frame, right_eye_coords, glasses_size, (0, 0, 0), -1)

        #Draw lines between glasses and ears
        frame = cv2.line(frame, left_ear_coords, left_eye_coords, (0, 0, 0), 2)
        frame = cv2.line(frame, right_ear_coords, right_eye_coords, (0, 0, 0), 2)
        frame = cv2.line(frame, left_eye_coords, right_eye_coords, (0, 0, 0), 4)

    return frame

def main(video_iterator, model):
    for frame in video_iterator:
        #inference pose model
        frame, results = model(frame)
        #Draw glasses on the face
        draw_glasses(frame, model)
        cv2.imshow("frame", frame)

        #blacks out the background if pose detected
        if results.pose_landmarks:
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            frame = frame * condition

if __name__ == "__main__":
    # Parse args from command line
    args = parser.parse_args()
    video_name = args.video_name

    video_iterator = VideoIterator(0)
    model = PoseDetector(enable_segmentation=True)

    main(video_iterator, model)
