import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Rock-Paper-Scissors logic mapping
RPS_OUTCOMES = {
    ("Rock", "Scissors"): "Player 1 Wins!",
    ("Scissors", "Paper"): "Player 1 Wins!",
    ("Paper", "Rock"): "Player 1 Wins!",
    ("Scissors", "Rock"): "Player 2 Wins!",
    ("Paper", "Scissors"): "Player 2 Wins!",
    ("Rock", "Paper"): "Player 2 Wins!",
}

# Function to detect gestures based on finger states
def detect_gesture(finger_states):
    if all(state == 0 for state in finger_states):
        return "Paper"
    elif all(state == 1 for state in finger_states):
        return "Rock"
    elif finger_states[1] == 0 and finger_states[2] == 0:  # Index and Middle fingers closed
        return "Scissors"
    return None

# Start capturing video
cap = cv2.VideoCapture(0)
scores = [0, 0]  # [Player 1 score, Player 2 score]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    mid_x = w // 2  # Middle line x-coordinate

    # Draw a dividing line in the middle of the screen
    cv2.line(frame, (mid_x, 0), (mid_x, h), (255, 255, 255), 2)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gestures = [None, None]  # [Player 1 gesture, Player 2 gesture]

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Determine which side the hand is on
            wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * w
            player = 0 if wrist_x < mid_x else 1  # 0 for Player 1 (left), 1 for Player 2 (right)

            # Detect finger states
            finger_states = []
            palm_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * h

            for finger_tip in [
                mp_hands.HandLandmark.THUMB_TIP,
                mp_hands.HandLandmark.INDEX_FINGER_TIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp_hands.HandLandmark.RING_FINGER_TIP,
                mp_hands.HandLandmark.PINKY_TIP,
            ]:
                finger_y = hand_landmarks.landmark[finger_tip].y * h
                finger_states.append(1 if finger_y < palm_y else 0)

            # Detect gesture and assign to the player
            gesture = detect_gesture(finger_states)
            gestures[player] = gesture

            # Display detected gesture on the player's side
            x_pos = 10 if player == 0 else mid_x + 10
            cv2.putText(
                frame, f"Player {player + 1}: {gesture or 'None'}", (x_pos, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

    # Determine the winner and update scores
    if gestures[0] and gestures[1]:
        if gestures[0] == gestures[1]:
            result_text = "It's a Draw!"
        else:
            result_text = RPS_OUTCOMES.get((gestures[0], gestures[1]), "Error")
            if "Player 1" in result_text:
                scores[0] += 1
            elif "Player 2" in result_text:
                scores[1] += 1

        # Display result in the center
        cv2.putText(frame, result_text, (mid_x - 150, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Display scores
    cv2.putText(frame, f"Player 1: {scores[0]}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Player 2: {scores[1]}", (mid_x + 10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("Rock Paper Scissors - Two Players", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
