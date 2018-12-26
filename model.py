# Script to create and train the model
from reader import read_csv
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

X_train, y_train = read_csv('dataset/driving_log.csv')

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, epochs=1, shuffle=True)

model.save('model.h5')
