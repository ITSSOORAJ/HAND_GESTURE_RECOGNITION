import pickle

import cv2
import mediapipe as mp
import numpy as np
import time

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
word=[]
last_predicted_character = None
last_prediction_time = 0
prediction_cooldown = 1

labels_dict = {0: 'A', 1: 'B', 2: 'C',3:'H',4:'I',5:'Y',6:'L',7:'V',8:'F'}
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10
        

        prediction = model.predict([np.asarray(data_aux)])
        prediction_proba = model.predict_proba([np.asarray(data_aux)])
        max_proba = max(prediction_proba[0])
        print(max_proba)
        predicted_label = int(prediction[0])
        
        if max_proba < 0.7:
            predicted_character = "Unknown"
        else:
            predicted_character = labels_dict[predicted_label]
        current_time = time.time()
        if predicted_character != "Unknown" and predicted_character != last_predicted_character:
            # Append the predicted character to the word list
            if current_time - last_prediction_time > prediction_cooldown:
                word.append(predicted_character)
                last_prediction_time = current_time
                last_predicted_character = predicted_character
        # Draw a rectangle around the hand and show the predicted character
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # Show the word that has been built so far
        word_display = "".join(word)
        cv2.putText(frame, word_display, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()