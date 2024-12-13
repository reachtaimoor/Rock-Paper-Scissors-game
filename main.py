import cv2
from capture_finger_rps_callback import process_hand_gestures

# Start capturing video
cap = cv2.VideoCapture(0)

# Define the crop regions (x, y, width, height)
crop_regions = [
    (0, 0, 320, 480),  # Top-left
    (320, 0, 320, 480),  # Top-right
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)

    # Process each crop region
    for (x, y, w, h) in crop_regions:
        crop_frame = frame[y:y+h, x:x+w]
        processed_frame, prediction = process_hand_gestures(crop_frame)  # Capture both processed frame and prediction
        
        # Add the prediction text at the bottom of the cropped area
        text_position = (x + 10, y + h - 10)  # Position for the text
        cv2.putText(frame, prediction, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Place the processed frame back into the original frame
        frame[y:y+h, x:x+w] = processed_frame  

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()