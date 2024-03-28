import streamlit as st
import cv2
import mediapipe as mp
import pyautogui
import math

st.success(f'Welcome {st.session_state["user"]}')

st.title('Virtual Mouse')

st.divider()

st.subheader('Please click on `capture` to start and click on `close` to close')


def capture():
    st.session_state.close = True

    cap = cv2.VideoCapture(0)
    hand_detector = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,
                                             min_tracking_confidence=0.5)
    drawing_utils = mp.solutions.drawing_utils
    screen_width, screen_height = pyautogui.size()

    while st.session_state.close:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = hand_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks

        if hands:
            for hand in hands:
                drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)

                # Detect index and thumb fingers
                index_finger = hand.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_finger = hand.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]

                # Convert normalized coordinates to pixel coordinates
                index_x = int(index_finger.x * frame_width)
                index_y = int(index_finger.y * frame_height)
                thumb_x = int(thumb_finger.x * frame_width)
                thumb_y = int(thumb_finger.y * frame_height)

                # Draw lines between landmarks
                cv2.line(frame, (index_x, index_y), (thumb_x, thumb_y), (255, 0, 0), 2)

                # Update cursor position
                cursor_x = screen_width * index_finger.x
                cursor_y = screen_height * index_finger.y
                pyautogui.moveTo(cursor_x, cursor_y)

                # Calculate distance between index and thumb finger
                distance = math.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)

                # Trigger click action when index finger meets thumb finger
                if distance < 30:  # Adjust the threshold as needed
                    print('click')
                    pyautogui.click()

        cv2.imshow('VM', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


capture = st.button('capture', on_click=capture)

close = st.button('Close')

if close:
    st.session_state.close = False
