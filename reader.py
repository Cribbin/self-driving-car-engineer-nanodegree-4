import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn

def generator(samples, batch_size=32):
    """Assumes all image paths in samples are correct absolute paths."""
    num_samples = len(samples)
    
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image_path = batch_sample[0]
                center_image = cv2.imread(center_image_path)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            X = np.array(images)
            y = np.array(angles)
            yield sklearn.utils.shuffle(X, y)
    

def samples_from_csvs(csv_paths, validation_split):
    """Returns a split of train and validation samples."""
    samples = []
    
    for p in csv_paths:
        with open(p) as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None) # Skip header
            samples += [line for line in reader]
            
    return train_test_split(samples, test_size=validation_split)


if __name__ == '__main__':
    samples = samples_from_csvs(['dataset/driving_log.csv'], 0.2)
    gen = generator(samples[0])
    next(gen)
    next(gen)