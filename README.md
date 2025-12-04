# Boxing Dynamics

Boxing Dynamics is a Python pipeline for analyzing the kinematics of a person boxing. The input to the pipeline is videos of person throwing a variety of punches. The pipieline will automatically analyze the video using Google's Pose Landmark Detection software mediapipe and detect the motion of 33 keypoints trhought the video. The keypoints are defined as follows: 

* 0 - nose
* 1 - left eye (inner)
* 2 - left eye
* 3 - left eye (outer)
* 4 - right eye (inner)
* 5 - right eye
* 6 - right eye (outer)
* 7 - left ear
* 8 - right ear
* 9 - mouth (left)
* 10 - mouth (right)
* 11 - left shoulder
* 12 - right shoulder
* 13 - left elbow
* 14 - right elbow
* 15 - left wrist
* 16 - right wrist
* 17 - left pinky
* 18 - right pinky
* 19 - left index
* 20 - right index
* 21 - left thumb
* 22 - right thumb
* 23 - left hip
* 24 - right hip
* 25 - left knee
* 26 - right knee
* 27 - left ankle
* 28 - right ankle
* 29 - left heel
* 30 - right heel
* 31 - left foot index
* 32 - right foot index


The output from the pipeline will be the input video with measured hip and shoulder rotations, as well as the hand velocity of both the left and right hands. These measurements directly correlate to an overall punch force metric. 



# Usage
In order to run the pipeline run the following in terminal : 

```
python3 main.py media/realspeed/jab.MP4
```


To run the pipeline and launch a debugger on an error:
```
python3 -m pdb -c continue main.py media/realspeed/jab.MP4
```


# Pipeline
The pipeline consist of 8 stages total. 
1. Video Loader
2. Landmark Extraction 
3. Kinematics Computations
4. Angular Kinematics Computations
5. Compute Boxing Metrics
6. Adding Force Arrows
7. Fusion of Boxing Metrics to Video
8. Output Video

