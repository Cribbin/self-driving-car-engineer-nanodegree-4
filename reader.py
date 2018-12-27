import csv
import cv2
import numpy as np

def read_csv(file_path):
    with open(file_path) as csvfile:
        lines = [line for line in csv.reader(csvfile)]

    images = []
    measurements = []
    for line in lines[1:]:
        center_path = 'dataset/' + line[0]
        center_image = cv2.imread(center_path)
        left_path = 'dataset/' + line[1]
        right_path = 'dataset/' + line[2]
        
        steering = float(line[3])
        throttle = float(line[4])
        brake = float(line[5])
        speed = float(line[6])
        
        images.append(center_image)
        measurements.append(steering)
        
        # Add flipped image
        images.append(cv2.flip(center_image, 1))
        measurements.append(-steering)
        
    

    return np.array(images), np.array(measurements)


if __name__ == '__main__':
    file_path = 'dataset/driving_log.csv'
    X_train, y_train = read_csv(file_path)