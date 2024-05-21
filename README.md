# Machine-learning_Sleep-detection-Sensor
 Abstract
Distracted driving, driving under influence, unskilled driving are well-known 
hazards, but few people think twice about getting behind the wheel when feeling 
drowsy. Each year around 100,000 traffic crashes occur because of sleep disorder 
in the United States. Following the former, researches based on computer vision 
application has devised high pixel cameras for sleep detection. In this work, a 
sensor is developed using deep learning to determine the drowsiness of driver by 
continuosly observing his/her eye states. To detect the eye region, OpenCV haar 
cascade classifier is used. Convolution neural network is built to extract the eye 
feature from dynamically captured camera frames. Fully connected softmax layer 
of CNN classifying the driver as sleep or non-sleep. This system alerts the driver 
with buzzer if sensor detects him/her in sleep mood. The model is caliberated on 
collected dataset consist of open/close eye images under different lighting 
conditions and clocked the accuracy of more than 90% during testing phase.

For DATASET, refer to link below:
https://data-flair.training/blogs/python-project-driver-drowsiness-detection-system/
