import mediapipe as np
import cv2

vid = cv2.VideoCapture(0)
np_pose = np.solutions.pose
np_draw = np.solutions.drawing_utils
np_hands = np.solutions.hands
hands = np_hands.Hands()

#img = cv2.imread("stalin.png")
#cv2.imshow("Student", img)
with np_pose.Pose() as pose:
    while True:
        ret, frame_hand = vid.read()
        frame_arm = frame_hand.copy()
        isRaised = False
        results = pose.process(frame_arm)
        hands_detected = hands.process(frame_hand)
        if results.pose_landmarks is not None:
            fred_dot = np_draw.DrawingSpec(color=(0, 0, 255), thickness=10, circle_radius=1)
            np_draw.draw_landmarks(frame_arm, results.pose_landmarks, np_pose.POSE_CONNECTIONS)
            arm_landmarks = results.pose_landmarks.landmark
            if arm_landmarks[16].y < arm_landmarks[14].y < arm_landmarks[12].y:
                frame_arm = cv2.putText(frame_arm, 'Arm is raised', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,cv2.LINE_AA, False)
                isRaised = True

        if hands_detected.multi_hand_landmarks:
            for hand_landmarks in hands_detected.multi_hand_landmarks:
                np_draw.draw_landmarks(frame_hand, hand_landmarks, np_hands.HAND_CONNECTIONS)
        if hands_detected.multi_hand_landmarks is not None:
            for hand_landmarks in hands_detected.multi_hand_landmarks:
                thumb_tip = hand_landmarks.landmark[4].y
                thumb_mid = hand_landmarks.landmark[3].y
                pointer_tip = hand_landmarks.landmark[8].y
                pointer_base = hand_landmarks.landmark[7].y
                middle_tip = hand_landmarks.landmark[12].y
                middle_base = hand_landmarks.landmark[11].y
                ring_tip = hand_landmarks.landmark[16].y
                ring_base = hand_landmarks.landmark[15].y
                pink_tip = hand_landmarks.landmark[20].y
                pink_mid = hand_landmarks.landmark[18].y
            if pointer_tip < pointer_base and middle_tip < middle_base and ring_tip < ring_base and pink_tip >= pink_mid and thumb_tip >= middle_base:
                if isRaised:
                    frame_hand = cv2.putText(frame_hand, 'We have a volunteer as tribute!', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA, False)

        cv2.imshow("Hand", frame_hand)
        cv2.imshow("Arm", frame_arm)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
#red_dot = np_draw.draw_landmarks()
#np_draw.draw_detection(img, )

vid.release()
cv2.destroyAllWindows()