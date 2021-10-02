# import cv2
# import mediapipe as mp
# from vision_projects.utils import VideoIterator
# import time
#
# import time
# import cv2
# import mediapipe as mp
# from vision_projects.utils import VideoIterator
# import time
#
# mp_hands
# # Setup Models
# mp_hands = mp.solutions.hands
# model = mp_hands.Hands()
# mpDraw = mp.solutions.drawing_utils
#
# # Setup Webcam/Video
# frame_iter = VideoIterator(0, rgb=False)
#
# c_time = 0
# p_time = 0
#
#
# for frame in frame_iter:
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     height, width, _ = rgb_frame.shape
#
#     result = model.process(rgb_frame)
#
#     if result.multi_hand_landmarks:
#         for hand_lm in result.multi_hand_landmarks:
#             mp.solutions.drawing_utils.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
#             for idx, lm in enumerate(hand_lm.landmark):
#                 if idx == 8:
#                     centre_width = int(lm.x * width)
#                     centre_height = int(lm.y * height)
#                     print(idx, centre_width, centre_height)
#                     cv2.circle(frame, (centre_width, centre_height), 10, (0,0,255), cv2.FILLED)
#
#     c_time = time.time()
#     fps = 1 / (c_time - p_time)
#     p_time = c_time
#
#     cv2.putText(frame, f"FPS: {fps:.2f}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
#     cv2.imshow("frame", frame)
