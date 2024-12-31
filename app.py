import cv2 as cv
import mediapipe as mp

hands = mp.solutions.hands.Hands( # init detector
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mpDraw = mp.solutions.drawing_utils # init drawing tool

cam = cv.VideoCapture(1) # initialize camera


if not cam.isOpened(): # check camera
    print("Error: Could not open camera.")
    exit()

while True:
    ret, cvFrame = cam.read() 

    mpFormatFrame = cv.cvtColor(cvFrame, cv.COLOR_BGR2RGB) # change to RGB format since mp use RGB not BGR
    # mpFrame = mp.Image(image_format = mp.ImageFormat.SRGB, data = mpFormatFrame)

    results = hands.process(mpFormatFrame)

    if results.multi_hand_landmarks: # if hand detected, draw it
        for hand_landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(cvFrame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    cv.imshow("Camera", cvFrame) # creates display

    if cv.waitKey(1) == ord('q'): # press q to quit
        break

cam.release()
cv.destroyAllWindows()