import cv2
import mediapipe as mp
import numpy as np
import joblib  # For loading the model
import time  # For timing operations
import streamlit as st
import google.generativeai as genai

genai.configure(api_key='AIzaSyC245Sr1xeRQGAocIgyLZr51qa9fdNOUUE')

st.success(f'Welcome *{st.session_state["user"][0]}*')

st.title('Sign Language Translator')

st.divider()

st.subheader('Please click on `capture` to start and click on `close` to close')


def capture():
    st.session_state.close = True

    model = joblib.load('pages/sign_language_model.pkl')

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    # Initialize variables
    last_probable_letter = None
    start_time = None
    letters = []
    sign_detected = ""

    def calculate_ratios(landmarks):
        """Calculate distances between specific hand landmarks, similar to data collection phase."""
        ratios = []
        for i in range(5, 21, 4):
            for j in range(i + 1, 21, 4):
                d = np.linalg.norm(
                    np.array((landmarks[i].x, landmarks[i].y)) - np.array((landmarks[j].x, landmarks[j].y)))
                ratios.append(d)
        return np.array(ratios).reshape(1, -1)  # Reshape for a single sample

    def display_words(frame, words):
        if words:
            words_str = ' '.join(words)
            cv2.putText(frame, words_str, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cap = cv2.VideoCapture(0)

    while st.session_state.close:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                ratios = calculate_ratios(landmarks)

                # Predict sign
                prediction = model.predict(ratios)[0]  # Assuming the prediction returns an array, get the first item
                probability = model.predict_proba(ratios).max()  # Get the highest probability

                # Display the prediction on the frame
                text = f"Sign: {prediction}, Probability: {probability:.2f}"
                cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                print(letters)

                # Check if the current prediction is the same as the last one
                if prediction == last_probable_letter:
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= 0.1:
                        sign_detected = prediction
                        letters.append(sign_detected)
                        start_time = None
                else:
                    start_time = None
                    last_probable_letter = prediction
                    sign_detected = ""  # Reset sign_detected when a new sign is detected

                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Process letters to form words and display them on the frame
        display_words(frame, letters)

        st.session_state['letters'] = letters

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to break
            break


capture = st.button('capture', on_click=capture)

close = st.button('Close')

if close:
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(f"Purify the sentence: {st.session_state['letters']}")
    st.subheader(f'The sentence: {response.text}')
    st.session_state.close = False