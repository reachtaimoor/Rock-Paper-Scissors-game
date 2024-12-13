import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Set the distance threshold for proximity check
CLOSE_DISTANCE_THRESHOLD = 30  # pixels

def get_finger_and_palm_positions(hand_landmarks):
    """Extracts finger and palm positions from hand landmarks."""
    finger_positions = {
        "Thumb": hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
        "Index Finger": hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
        "Middle Finger": hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
        "Ring Finger": hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
        "Pinky": hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP],
    }
    
    palm_position = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    return finger_positions, palm_position

def draw_bounding_box(frame, hand_landmarks):
    """Draws a bounding box around specific hand landmarks."""
    points = []
    landmark_indices = [
        mp_hands.HandLandmark.WRIST,  # 0
        mp_hands.HandLandmark.THUMB_CMC,  # 1
        mp_hands.HandLandmark.INDEX_FINGER_MCP,  # 2
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,  # 5
        mp_hands.HandLandmark.RING_FINGER_MCP,  # 9
        mp_hands.HandLandmark.PINKY_MCP  # 13
    ]
    
    for index in landmark_indices:
        landmark = hand_landmarks.landmark[index]
        points.append((int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])))

    if points:
        x_coords, y_coords = zip(*points)
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        return (x_min, y_min, x_max, y_max)  # Return bounding box coordinates

def is_finger_close_to_box(finger_position, box_coords):
    """Check if the finger tip is close to the bounding box."""
    x_min, y_min, x_max, y_max = box_coords
    finger_x, finger_y = finger_position

    # Check proximity to the bounding box
    if (x_min - CLOSE_DISTANCE_THRESHOLD <= finger_x <= x_max + CLOSE_DISTANCE_THRESHOLD and
        y_min - CLOSE_DISTANCE_THRESHOLD <= finger_y <= y_max + CLOSE_DISTANCE_THRESHOLD):
        return True
    return False

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            box_coords = draw_bounding_box(frame, hand_landmarks)

            finger_positions, palm_position = get_finger_and_palm_positions(hand_landmarks)

            # Initialize finger states as an array
            finger_states = [0, 0, 0, 0, 0]  # [Thumb, Index, Middle, Ring, Pinky]
            palm_y = int(palm_position.y * frame.shape[0])  # Get palm Y-coordinate for comparison

            finger_names = ["Thumb", "Index Finger", "Middle Finger", "Ring Finger", "Pinky"]

            for i, finger_name in enumerate(finger_names):
                landmark = finger_positions[finger_name]
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.putText(frame, finger_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Check if the finger is close to the bounding box
                if is_finger_close_to_box((x, y), box_coords):
                    cv2.putText(frame, f"{finger_name} Close ", (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    # Determine if the finger is open (1) or closed (0) only if close to the box
                    finger_states[i] = 1 if y < palm_y else 0  # Compare with palm Y-coordinate
                else:
                    finger_states[i] = 0  # Indicate that the finger is not close

            palm_x = int(palm_position.x * frame.shape[1])
            palm_y = int(palm_position.y * frame.shape[0])
            cv2.putText(frame, "Palm", (palm_x, palm_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Display the state of each finger in a small table on the right side
            y_offset = 30
            cv2.putText(frame, "Finger States:", (frame.shape[1] - 150, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            for i, state in enumerate(finger_states):
                cv2.putText(frame, f"{finger_names[i]}: {state}", (frame.shape[1] - 150, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

            # Determine and display Rock, Paper, or Scissors
            if all(state == 0 for state in finger_states):
                cv2.putText(frame, "Paper", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif all(state == 1 for state in finger_states):
                cv2.putText(frame, "Rock", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif finger_states[1] == 0 and finger_states[2] == 0 and finger_states[3] == 1 and finger_states[4] == 1 and finger_states[0] == 1:  # Index and Middle fingers closed
                cv2.putText(frame, "Scissors", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Finger and Palm Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()