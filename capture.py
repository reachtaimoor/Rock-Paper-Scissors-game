import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Create directories for storing images
os.makedirs('rock_images', exist_ok=True)
os.makedirs('paper_images', exist_ok=True)
os.makedirs('scissors_images', exist_ok=True)

cap = cv2.VideoCapture(0)
counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Logic to determine gesture based on landmarks
            # Example: Check if fingers are up or down to classify gesture
            # This is a simplified example; you may need to implement more complex logic
            fingers = [0] * 5  # Assuming 5 fingers
            for i in range(5):
                if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y:
                    fingers[i] = 1  # Finger is up
                else:
                    fingers[i] = 0  # Finger is down

            # Determine gesture
            if fingers == [0, 0, 0, 0, 0]:  # Rock
                cv2.putText(frame, "Rock", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.imwrite(f'rock_images/rock_{counter}.jpg', frame)
                counter += 1
            elif fingers == [1, 1, 1, 1, 1]:  # Paper
                cv2.putText(frame, "Paper", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.imwrite(f'paper_images/paper_{counter}.jpg', frame)
                counter += 1
            elif fingers == [1, 1, 0, 0, 0]:  # Scissors
                cv2.putText(frame, "Scissors", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.imwrite(f'scissors_images/scissors_{counter}.jpg', frame)
                counter += 1

    cv2.imshow("Rock Paper Scissors Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()