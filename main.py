import cv2
import numpy as np
import os

faceCasc = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
eyesCasc = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades + 'haarcascade_eye.xml'))

def face_detect(vidGray,frame):
    #making retangle around the face
    faceRect = faceCasc.detectMultiScale(vidGray ,1.3 ,5)
    #drawing the rectangle on the face
    for (x ,y ,w ,h) in faceRect:
        cv2.rectangle(frame, (x,y),(x+w ,y+h), (0,0,255), 3)
        #make the face in the rectangle black and white
        roi_gray = vidGray[y:y+h , x:x+w]
        #make the face in the rectangle coloured
        roi_color = frame[y:y+h , x:x+w]
        #find eyes in the rectangle
        eyes = eyesCasc.detectMultiScale(roi_gray, 1.1, 3)
        for (ex,ey,ew,eh) in eyes:
            #draws rectangle for the found eyes
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,0,255), 2)
    
    return frame
        






#take the video
vid = cv2.VideoCapture(0)
#while loop runs forever
while True:
    #reads the image
    _,frame = vid.read()
    #make the face gray
    vidGray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #sending parameters to the function
    vidfunc = face_detect(vidGray,frame)
    #display
    cv2.imshow('Face Detector',vidfunc)
    #if you press a key from keyboard loop stops
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#close camera
vid.release()
cv2.destroyAllWindows()


