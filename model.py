# Script to create and train the model
from reader import samples_from_csvs, generator
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Conv2D
from keras.layers.pooling import MaxPooling2D

# Train and validation samples
samples = samples_from_csvs(['dataset/driving_log.csv',
                             'dataset/forward/driving_log.csv',
                             'dataset/reverse/driving_log.csv',
                             'dataset/edges/driving_log.csv',
                             'dataset/correctionlap/driving_log.csv',
                             'dataset/cornerlap/driving_log.csv'],
                            validation_split=0.2)

# Will create batches of 32 train/validation samples
# Note: The real size is 96 images, as for each sample, it takes the left, center, and right images
train_generator = generator(samples[0], batch_size=32)
validation_generator = generator(samples[1], batch_size=32)

# NN implementation of https://devblogs.nvidia.com/deep-learning-self-driving-cars/
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5)) # Additional dropout layer to prevent overfitting
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Adam optimizer
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
                    steps_per_epoch=len(samples[0]),
                    validation_data=validation_generator,
                    validation_steps=len(samples[1]),
                    epochs=2)

model.save('model.h5')
