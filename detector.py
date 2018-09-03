# Step 4: Run Face Recognition Application

import cv2
import time
import numpy as np 
import sqlite3
import os
conn = sqlite3.connect('database.db')
c = conn.cursor()
fname = "recognizer/trainingData.yml"
if not os.path.isfile(fname):
    print("Please train the data first")
    exit(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(fname)


while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        font                   = cv2.FONT_HERSHEY_DUPLEX
        bottomLeftCornerOfText = (20,50)
        fontScale              = 1
        fontColor              = (0,255,0)
        lineType               = 1

        cv2.putText(img,'Smart Door', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
        ids,conf = recognizer.predict(gray[y:y+h,x:x+w])
        c.execute("select name from users where id = (?);", (ids,))
        result = c.fetchall()
        name = result[0][0]
        print(name)
        print("conf: ", conf)
        if conf < 50:
            cv2.putText(img, name, (x+2,y+h-5), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0),2)
            print("DETECTED: ", name)
        else:
            cv2.putText(img, 'Unknown', (x+2,y+h-5), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255),1)
        cv2.imshow('Smart Door',img)

    k = cv2.waitKey(30) & 0xff

    if k == 27:
        print("break")
        break

cap.release()
cv2.destroyAllWindows()