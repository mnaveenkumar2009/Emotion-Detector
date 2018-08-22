from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import numpy as np

model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(48,48,1)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.load_weights('python_model/emotion_model02.h5')

import cv2

face_cascade = cv2.CascadeClassifier('python_model/face.xml')

cap = cv2.VideoCapture(0)
 
# loop runs if capturing has been initialized.
while 1: 
 
    # reads frames from a camera
    ret, img = cap.read() 
 
    # convert to gray scale of each frames
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    # Detects faces of different sizes in the input image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        # To draw a rectangle in a face 
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv2.imshow('img',roi_gray)
        roi_gray = cv2.resize(roi_gray  , (48 , 48))
        roi_gray.resize(1,48,48,1)
        for i in range(48):
            for j in range(48):
                roi_gray[0][i][j][0] = roi_gray[0][j][i][0]
        xxx = model.predict(roi_gray, batch_size=None, verbose=0, steps=None)
        emotion = 'angry'
        print np.argmin(xxx)
        print xxx
        cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('img',img)
    # Wait for Esc key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
 
# Close the window
cap.release()

cv2.destroyAllWindows() 