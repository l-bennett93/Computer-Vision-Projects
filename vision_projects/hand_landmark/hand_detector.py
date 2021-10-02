import cv2
import mediapipe as mp
import time

from vision_projects.utils import VideoIterator


class HandDetector:
    def __init__(self,
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        draw=True
    ):
    """
    Hand Detection model.
    Provides landmark and drawing capabilites to detect hand landmarks in videos

    Parameters
    ----------
    static_image_mode: bool
        if True will track frames to reduce overhead. If false, will always detect

    max_num_hands=int
        Maximum number of hands to track in one frame

    min_detection_confidence=0.5
        The percentage confidence required to detect an object

    min_tracking_confidence=0.5
        The percentage confidence required to maintain a detection

    draw=True
        If true, will draw the lankmarks detected n the frame

    """

        self.model = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.draw=draw

    def __call__(self, frame):
        """
        Finds the hand landmarks for a given frame.
        If self.draw is true, it will also draw the landmarks on the image

        Parameters
        ----------
        frame: np.ndarray
            input BGR frame

        Returns
        -------
        frame:
            output BGR frame including landmarks, if draw is True

        landmarks: mediapipeline.multi_hand_landmarks
            Landmarks for the hands detected. Will be limited
            by the max_num_hands defined in the init

        """

        rgb_frame=self._preprocess(frame)
        result = self.model.process(rgb_frame)

        #Setup height and width if not setup
        if not hasattr(self, "height"):
            self.height, self.width, _ = frame.shape

        if result.multi_hand_landmarks:
            for hand_lm in result.multi_hand_landmarks:
                if self.draw:
                    frame = self._draw(frame, hand_lm)

        self._last_result = result
        return frame, result.multi_hand_landmarks

    @staticmethod
    def _preprocess(frame):
        """
        Converts the frame from BGR to RGB

        Parameters
        ----------
        bgr_frame: np.ndarray
            bgr image
        Returns
        -------
        rgb_frame: np.ndarray
            rgb image

        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb_frame

    def _draw(self, frame, hand_lm):
        """
        Draws the hand landmarks on the frame and returns the frame

        Parameters
        ----------
        frame: np.ndarray
            input frame to draw on

        hand_lm: mediapipe.landmarks
            The landmarks to draw on the image

        Returns
        -------
        frame: np.ndarray
            frame with the landmarks drawn.

        """
        mp.solutions.drawing_utils.draw_landmarks(frame, hand_lm, mp.solutions.hands.HAND_CONNECTIONS)

        for idx, lm in enumerate(hand_lm.landmark):
            centre_width = int(lm.x * self.width)
            centre_height = int(lm.y * self.height)
            cv2.circle(frame, (centre_width, centre_height), 10, (0,0,255), cv2.FILLED)

        return frame

    def find_position(self, hand_number, landmark_idx):
        """
        Finds a lanmarks for a given hand and landmark id

        Parameters
        ----------
        hand_number: int
            The index of the hand to grab

        landmark_idx: int
            The index of the landmark to grab

        Returns
        -------
        landmark_position: tuple
            (x,y) tuple of the position of the landmark

        """
        multi_lmks = self._last_result.multi_hand_landmarks

        if multi_lmks:
            if len(multi_lmks) >= hand_number:
                hand_lm = multi_lmks[hand_number]
        else:
            hand_lm = None

        if hand_lm:
            hand_lm = dict(enumerate(hand_lm.landmark))
            lm = hand_lm.get(landmark_idx)
            if lm:
                centre_width = int(lm.x * self.width)
                centre_height = int(lm.y * self.height)
            return (centre_width, centre_height)
        return None
