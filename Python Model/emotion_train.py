import numpy as np
import pandas
from sklearn.model_selection import train_test_split

filename = 'dataset/train.csv'
data = pandas.read_csv(filename)
# print(data.head)
np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils


Y = data.Emotion
X = data.drop('Emotion', axis=1)
X.Pixels = X.Pixels[0]
XX = [[] for x in range(4178)]
for i in range(4178):
    XX[i] = np.reshape(X.Pixels[i].split(' '),(48,48,1))
X_train = XX[:2800]
X_test = XX[2801:]
Y_train = Y[:2800]
Y_test = Y[2801:]



X_train = np.array(X_train)
X_test = np.array(X_test)
X_train.astype(np.float)
X_test.astype(np.float)
Y_train = np_utils.to_categorical(Y_train, 7)
Y_test = np_utils.to_categorical(Y_test, 7)
# X_train /= 255.0
# X_test /= 255.0
# np.divide(X_train,255)
# np.divide(X_test,255)

 	
model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(48,48,1)))
print model.output_shape
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=10, verbose=1)

 	
score = model.evaluate(X_test, Y_test, verbose=0)

print score

model.save('emotion.h5')