import cv2
import mediapipe as mp
import pyautogui
import numpy as np

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()
roi_x = 200  # X-coordinate of the top-left corner of the ROI
roi_y = 150  # Y-coordinate of the top-left corner of the ROI
roi_width = 190  # Width of the ROI
roi_height = 90  # Height of the ROI


while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    if landmark_points:
        landmarks = landmark_points[0].landmark

        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 2, (255, 255, 255))
            if id==1:
                screen_x = int(screen_w * landmark.x * 1.25)
                screen_y = int(screen_h * landmark.y * 1.25)
                pyautogui.moveTo(screen_x,screen_y)
            left = [landmarks[145], landmarks[159]]
            for landmark in left:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255))
            if (left[0].y - left[1].y) < 0.005:
                pyautogui.click()
                pyautogui.sleep(1)
    cv2.imshow("Eye controlled mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
