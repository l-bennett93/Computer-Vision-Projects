import cv2

class VideoIterator:
    def __init__(self, video_name, rgb=False):
        """

        Parameters
        ----------
        video_name: str
            file path of the video to be processed.
            If 0, will use the web camera

        rgb: bool
            If RGB is True, then the image will be in RBG, otherwise the image
            will be BGR
        """

        self.video_name = video_name
        self.rgb = rgb

    def __iter__(self):
        """
        Opens the video file and iterates over the file until complete

        Yields
        ------
        frame: np.ndarray
            3d np.array containing an image from the video
        """
        cap = cv2.VideoCapture(self.video_name)
        assert cap.isOpened(), "Video cannot be captured"

        try:
            while True:
                ret, frame = cap.read()
                if ret:
                    # Convert the frame from BGR to RGB if defined
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if self.rgb else frame
                    yield frame

                if cv2.waitKey(1) & 0xff == ord("q"):
                    break

        finally:
            cap.release()
