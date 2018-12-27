# Script to create and train the model
from reader import samples_from_csvs, generator
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Conv2D
from keras.layers.pooling import MaxPooling2D

samples = samples_from_csvs(['dataset/driving_log.csv'],
                            validation_split=0.2)

train_generator = generator(samples[0], batch_size=32)
validation_generator = generator(samples[1], batch_size=32)

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    steps_per_epoch=len(samples[0]),
                    validation_data=validation_generator,
                    validation_steps=len(samples[1]),
                    epochs=1)
#model.fit(X_train, y_train, validation_split=0.2, epochs=2, shuffle=True)

model.save('model.h5')
