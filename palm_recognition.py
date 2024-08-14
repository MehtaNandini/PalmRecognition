import cv2
import mediapipe as mp

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe drawing utilities.
mp_drawing = mp.solutions.drawing_utils

# Start video capture from webcam.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect hands.
    results = hands.process(image)
    
    # Convert the image color back to BGR so it can be displayed properly in OpenCV.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Draw hand landmarks on the image.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Display the resulting image.
    cv2.imshow('Palm Recognition', image)
    
    # Exit on pressing 'q'.
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the webcam and close the window.
cap.release()
cv2.destroyAllWindows()