import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Function to get finger and palm positions
def get_finger_and_palm_positions(hand_landmarks):
    finger_positions = {
        "Thumb": hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP],
        "Index Finger": hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
        "Middle Finger": hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
        "Ring Finger": hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP],
        "Pinky": hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP],
    }
    
    # Palm position can be approximated by the wrist landmark
    palm_position = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    return finger_positions, palm_position

# Function to draw bounding box around the palm and immediate joints
def draw_bounding_box(frame, palm_position, finger_positions):
    # Collect points for bounding box
    points = []
    
    # Add palm position
    points.append((int(palm_position.x * frame.shape[1]), int(palm_position.y * frame.shape[0])))
    
    # Add finger joint positions
    for landmark in finger_positions.values():
        points.append((int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])))

    # Calculate bounding box
    if points:
        x_coords, y_coords = zip(*points)
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

# Start capturing video
cap = cv2.VideoCapture(0)

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

            # Get finger and palm positions
            finger_positions, palm_position = get_finger_and_palm_positions(hand_landmarks)

            # Display finger positions on the frame
            for finger_name, landmark in finger_positions.items():
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.putText(frame, finger_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Display palm position
            palm_x = int(palm_position.x * frame.shape[1])
            palm_y = int(palm_position.y * frame.shape[0])
            cv2.putText(frame, "Palm", (palm_x, palm_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Draw bounding box around the palm and immediate joints
            draw_bounding_box(frame, palm_position, finger_positions)

    # Display the frame
    cv2.imshow("Finger and Palm Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()