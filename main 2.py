import cv2
import time
from capture_finger_rps_callback import process_hand_gestures

# Global variable to store predictions
predictions = ["", ""]  # Index 0 for player 1, index 1 for player 2

# Function to determine the winner
def determine_winner(player1, player2):
    if player1 == player2:
        return "Draw"
    elif (player1 == "Rock" and player2 == "Scissors") or \
         (player1 == "Scissors" and player2 == "Paper") or \
         (player1 == "Paper" and player2 == "Rock"):
        return "Left Player Wins!"
    else:
        return "Right Player Wins!"

# Start capturing video
cap = cv2.VideoCapture(0)

# Define the crop regions (x, y, width, height)
crop_regions = [
    (0, 0, 320, 480),  # Top-left (Player 1)
    (320, 0, 320, 480),  # Top-right (Player 2)
]

while True:
    # Overlay messages
    overlay_messages = [
        "This is a Game",
        "Show your hand gesture"
    ]
    
    # Display initial message
    start_time = time.time()
    predictions = ["", ""]  # Reset predictions for the new game

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)

        # Process each crop region
        for index, (x, y, w, h) in enumerate(crop_regions):
            crop_frame = frame[y:y+h, x:x+w]
            processed_frame, prediction = process_hand_gestures(crop_frame)  # Capture both processed frame and prediction
            
            # Save the prediction in the global variable
            predictions[index] = prediction

            # Place the processed frame back into the original frame
            frame[y:y+h, x:x+w] = processed_frame  

        # Overlay the messages
        elapsed_time = time.time() - start_time

        if elapsed_time < 1:
            message = overlay_messages[0]
        elif elapsed_time < 2:
            message = overlay_messages[1]
        elif elapsed_time < 7:  # 5 seconds timer after showing the hand gesture
            message = "Time Left: {}".format(max(0, 7 - int(elapsed_time)))
        else:
            # After 5 seconds, determine the winner and display the result
            player1_prediction = predictions[0]
            player2_prediction = predictions[1]
            message = determine_winner(player1_prediction, player2_prediction)
            cv2.putText(frame, message, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Break the inner loop after showing the result
            if elapsed_time > 10:  # Show the result for 3 seconds
                break

        # Display the overlay message on the frame
        cv2.putText(frame, message, (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Rock Paper Scissors Game", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()  # Exit the program if 'q' is pressed

    # Reset the timer for the next game
    start_time = time.time()

cap.release()
cv2.destroyAllWindows()