import mediapipe as np
import cv2

vid = cv2.VideoCapture(0)
np_pose = np.solutions.pose
np_draw = np.solutions.drawing_utils
np_hands = np.solutions.hands
hands = np_hands.Hands()

while True:
    ret, frame_arm = vid.read()
    with np_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(frame_arm)
    if results is not None:
        fred_dot = np_draw.DrawingSpec(color=(0, 0, 255), thickness=10, circle_radius=1)
    #np_draw.draw_landmarks(frame_arm, landmark_list=results.pose_landmarks, landmark_drawing_spec= red_dot)
    np_draw.draw_landmarks(frame_arm, results.pose_landmarks, np_pose.POSE_CONNECTIONS)

    cv2.imshow("Arm", frame_arm)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.waitKey(0)
cv2.destroyAllWindows()