import cv2

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150) # BRIGHTNESS
while True:
    success, img = cap.read()
    img =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF in [ord('q'),ord('Q'),27]:
        break