import cv2;import pyttsx3;from time import time
thres = 0.50 # Threshold to detect object
start = time()
cap = cv2.VideoCapture(1)
cap.set(3,720)
cap.set(4,360)
cap.set(10,100)
engine = pyttsx3.init('sapi5')
engine.setProperty('speed',50)
voices = engine.getProperty('voices')
engine.setProperty('voice',voices[1].id)
classNames= []
classFile = 'coco.names'
with open(classFile) as f:
    classNames = f.read().rstrip('\n').split('\n')
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            if classNames[classId-1]!=" ":
                cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                cv2.putText(img,classNames[classId-1].upper(),(box[0],box[1]),
                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                engine.say(classNames[classId-1])
                engine.runAndWait()
    cv2.imshow("Output",img)
    if cv2.waitKey(1) == ord('q'):
        break