# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* reader.py respondible for reading in the training/validation data and serving as a generator
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

All the code for the model can be found in `model.py`.

The model I chose is an implementation of this [NVidia Self Driving Car Model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/). Its structure can be seen on lines 22-46. I will go into some more detail here.

First the data is normalized and mean centered using a Kera lambda layer.

The image is then cropped using the Keras Cropping2D layer, to reduce irrelavant pixels of sky/car.

The image is then passed through 5 convolution layers. These are all activated with a ReLU function to introduce nonlinearity to the model.

Next the output of the final convolution layer is flattened, and passed through a 4 fully connected layers. The last one is the steering angle output. There is also a single dropout layer after the first fully connected layer to reduce overfitting.

The model is trained for 2 epochs, and mean square error is reduced using an adam optimizer.

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layers in order to reduce overfitting (model.py line 35). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 9-14). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 41).

Epoch number is set to two. I noticed the training accuracy to start to fluctuate after the second epoch, indicating overfitting.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the following training data:
* Default training data from Udacity
* Two forward laps of the track (Center lane driving)
* Two reverse laps of the track (Allowing the model to generalize, and not to have a bias to one bend)
* One lap focusing on recovering from veering off to the side (I did not record the car driving to the sides, only the recovery)
* One lap recording only doing perfect corners to avoid a straight bias, as corners were difficult for the model to maneuver.
* One additional dataset of particularly dangerous scenarios (e.g. facing directly into a wall).

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

After directly implementing the NVidia model mentioned previously, as seen in the Udacity videos. I was able to get a low mean squared error on the training set, but my validation set remained high. It was decreasing after each epoch, but not at nearly the same rate as the training set.

At this point, I had generated samples from using the simulator on the workspace GPU. I tried testing on the simulator, and the car ran decently well, but failed to take certain corners.

My solution to this was to record tricky corners to allow the model to learn from these examples. I tried using trial and error. However, this approach did not work. I realized that I was overfitting the model to the track, and I shouldn't have used was is essentially the testing set to influence the inputs to my model.

So I scraped all the training data and downloaded the simulator to my machine. I found controlling the car difficult on the workspace, and figured that maybe the training data wasn't giving ideal inputs because of this. Therefore I generated training data locally on my machine (See earlier in the report what the different scenarios were).

I fed this into the model, and it worked much better. However, it still wasn't perfect. I decided to try and feed multiple camera images in while training, and that finally resulted in a model which could navigate the track. This allowed the model to generalize the data better, and to prevent overfitting.

#### 2. Final Model Architecture

The model I chose is an implementation of this [NVidia Self Driving Car Model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/). Its structure can be seen on lines 22-46. I will go into some more detail here.

First the data is normalized and mean centered using a Kera lambda layer.

The image is then cropped using the Keras Cropping2D layer, to reduce irrelavant pixels of sky/car.

The image is then passed through 5 convolution layers.
1. 24 5x5 filters with stride 2
2. 36 5x5 filters with stride 2
3. 48 5x5 filters with stride 2
4. 64 3x3 filters with stride 1
5. 64 3x3 filters with stride 1

These are all activated with a ReLU function to introduce nonlinearity to the model.

Next the output of the final convolution layer is flattened, and passed through a 4 fully connected layers. The last one is the steering angle output. There is also a single dropout layer after the first fully connected layer to reduce overfitting.

The model is trained for 2 epochs, and mean square error is reduced using an adam optimizer.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
