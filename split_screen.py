import cv2
from hand_gesture_detector import HandGestureDetector

def main():
    # Create an instance of the HandGestureDetector
    detector = HandGestureDetector()

    # Start capturing video from a single camera
    cap = cv2.VideoCapture(0)  # Change index if necessary

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video.")
            break

        # Split the frame into two halves
        height, width, _ = frame.shape
        left_half = frame[:, :width // 2]  # Left half of the frame
        right_half = frame[:, width // 2:]  # Right half of the frame

        # Process each half for gesture detection
        gesture_left, processed_frame_left = detector.process_frame(left_half)
        gesture_right, processed_frame_right = detector.process_frame(right_half)

        # Combine the processed frames back into one
        combined_frame = cv2.hconcat([processed_frame_left, processed_frame_right])

        # Display gestures on the combined frame
        if gesture_left:
            cv2.putText(combined_frame, f"Player 1: {gesture_left}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if gesture_right:
            cv2.putText(combined_frame, f"Player 2: {gesture_right}", (processed_frame_left.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the combined frame
        cv2.imshow("Split Screen - Player 1 (Left) | Player 2 (Right)", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()