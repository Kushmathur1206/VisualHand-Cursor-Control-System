# Import Libraries
import cv2
import time
import mediapipe as mp
import pyautogui
import math

mp_drawing = mp.solutions.drawing_utils
cam = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)
screen_w, screen_h = pyautogui.size()

prev_x, prev_y = -1, -1
movement_threshold = 5
i=0
while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)

    # Converting the from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Converting back the RGB image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = hands.process(image)
    landmark_points = results.multi_hand_landmarks
    frame_h, frame_w, _ = frame.shape
    hand_closed = False
    # If hand landmarks are detected
    if landmark_points:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 1, (255,255, 255),-1)
            if id == 8:
                screen_x = (screen_w * landmark.x)
                screen_y = (screen_h * landmark.y)
                if math.dist((prev_x, prev_y), (x, y)) > movement_threshold:
                    pyautogui.moveTo(screen_x, screen_y)
                prev_x, prev_y = x, y
            if id == 4:  # Thumb
                thumb_x, thumb_y = x, y
            if id == 8:  # Index finger
                index_x, index_y = x, y

                # Calculate the distance between thumb and index finger
                dist_thumb_index = math.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)

                # Set hand_closed flag if distance is below a threshold
                if dist_thumb_index < 14:  # Adjust threshold as needed
                    hand_closed = True
                else:
                    hand_closed = False

            # Perform action when hand is closed
            if hand_closed:
                print("closed ",i)
                i=i+1
                hand_closed = False
                pyautogui.click()
                pyautogui.sleep(1)
    # Displaying FPS on the image
    # Display the resulting image
    cv2.imshow("Facial and Hand Landmarks", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

