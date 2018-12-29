import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn

def generator(samples, batch_size=32):
    """
    Acts as a co-routine to create mini-batches of the samples list provided.
    Assumes all image paths in samples are correct absolute paths.
    Yields two numpy arrays of length batch_size*3:
        X: images from cameras
        y: steering angle taken
    """
    num_samples = len(samples)

    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            # Iterates through the samples in batches of batch_size (default 32)
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                correction = 0.2 # How much steering needed for right and left images
                
                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction
                right_angle = center_angle - correction
                
                center_image = cv2.imread(batch_sample[0])
                left_image = cv2.imread(batch_sample[1])
                right_image = cv2.imread(batch_sample[2])
                
                # Add the three camera images and their steering angles to lists
                images.append(center_image)
                angles.append(center_angle)
                
                images.append(left_image)
                angles.append(left_angle)
                
                images.append(right_image)
                angles.append(right_angle)

            # Convert the lists to numpy arrays
            X = np.array(images)
            y = np.array(angles)
            
            yield sklearn.utils.shuffle(X, y)
    

def samples_from_csvs(csv_paths, validation_split):
    """Returns a split of train and validation samples read in from the csv paths."""
    samples = []
    
    for p in csv_paths:
        with open(p) as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None) # Skip header
            samples += [line for line in reader]
            
    return train_test_split(samples, test_size=validation_split)