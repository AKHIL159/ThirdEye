import cv2
face_cascade = cv2.CascadeClassifier('C:/Users/sk503/Downloads/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(1)
while True:
    ret,img=cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    cv2.imshow('img',img)
    if cv2.waitKey(1) in [ord("q"),27]: # this can be exitted by pressing "q" key
        break
