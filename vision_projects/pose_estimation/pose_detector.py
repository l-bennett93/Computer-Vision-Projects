import cv2
import mediapipe as mp


class PoseDetector:
    def __init__(
        self,
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        draw=True
    ):

        """
        Pose Detection Model

        Parameters
        ----------
        static_image_mode: bool
            If set to false, the solution treats the input images as a video stream.
            It will try to detect the most prominent person in the very first images,
            and upon a successful detection further localizes the pose landmarks.
            In subsequent images, it then simply tracks those landmarks without
            invoking another detection until it loses track, on reducing computation
            and latency. If set to true, person detection runs every input image,
            ideal for processing a batch of static, possibly unrelated,
            images. Default to false.

        model_complexity: int
            Complexity of the pose landmark model: 0, 1 or 2.
            Landmark accuracy as well as inference latency generally go up
            with the model complexity. Default to 1.

        smooth_landmarks: bool
            If set to true, the solution filters pose landmarks across different
            input images to reduce jitter, but ignored if static_image_mode
            is also set to true. Default to true.

        enable_segmentation: bool
            If set to true, in addition to the pose landmarks the solution
            also generates the segmentation mask. Default to false.

        smooth_segmentation: bool
            If set to true, the solution filters segmentation masks across
            different input images to reduce jitter. Ignored if enable_segmentation
            is false or static_image_mode is true. Default to true.

        min_detection_confidence: float
            Minimum confidence value ([0.0, 1.0]) from the person-detection model
            for the detection to be considered successful. Default to 0.5.

        min_tracking_confidence: float
            Minimum confidence value ([0.0, 1.0]) from the landmark-tracking model
            for the pose landmarks to be considered tracked successfully, or otherwise
            person detection will be invoked automatically on the next input image.
            Setting it to a higher value can increase robustness of the solution,
            at the expense of a higher latency. Ignored if static_image_mode is true,
            where person detection simply runs on every image. Default to 0.5.

        draw: bool
            If True, will draw the landmarks on the image

        """

        self.model = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.draw=draw

    def __call__(self, frame):

            if not hasattr(self, "height"):
                self.height, self.width, _ = frame.shape

            rbg_frame = self._preprocess(frame)
            self.results = self.model.process(rbg_frame)

            if self.draw:
                self._draw_landmarks(frame, self.results.pose_landmarks)

            return frame, self.results

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

    def _draw_landmarks(self, frame, pose_landmarks):
        """
        Draws the pose detection landmarks on the person

        Parameters
        ----------
        frame: np.ndarray
            The frame to be drawn upon

        pose_landmarks: mediapose.solutions.pose.model.pose_landmarks
            The class containing the pose landmarks

        """
        mp.solutions.drawing_utils.draw_landmarks(frame, pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
        return frame

    def find_position(self, landmark_idx):
        """
        Finds the particular landmark from the detection using the landmark_idx

        Parameters
        ----------
        landmark_idx :int
            The integer representing the landmark to return.
            For more information regarding the landmark indexes see
            https://google.github.io/mediapipe/solutions/pose.html

        Returns
        -------
        pose_coords: tuple
            (x,y) coordinates of the particular landmark

        """
        if self.results.pose_landmarks:
            pose_landmarks = dict(enumerate(self.results.pose_landmarks.landmark))
            pose_lm = pose_landmarks.get(landmark_idx)

            if pose_lm:
                pose_coords = (int(pose_lm.x * self.width), int(pose_lm.y * self.height))
                return pose_coords

        return None
