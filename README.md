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

# Usage (PC)
In order to use the pipeline first clone the repository using : 
```
    git clone git@github.com:dhruvgargj5/boxing_dynamics.git
    cd boxing_dynamics
```

Next, create a virtual environment and install the requirements using : 
```
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
```

Finally, in order to run the pipeline use the following in terminal: 

```
    python main.py <path2Video/link2Video>
```

To run the pipeline and launch a debugger on an error:
```
python -m pdb -c continue main.py  <path2Video/link2Video>
```

# Usage (Linux/MAC)
In order to use the pipeline first clone the repository using : 
```
    git clone git@github.com:dhruvgargj5/boxing_dynamics.git
    cd boxing_dynamics
```

Next, create a virtual environment and install the requirements using : 
```
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
```

Finally, in order to run the pipeline use the following in terminal: 
```
    python3 main.py <path2Video/link2Video>
```

To run the pipeline and launch a debugger on an error:
```
python3 -m pdb -c continue main.py  <path2Video/link2Video>
```


# Pipeline
The pipeline consist of 6 stages total. 
1. Video Loader 
    > loads the video from a path and stores the video settings (name, fps, and scale factor)
2. Landmark Extraction 
    > uses the mediapipe pose detection software to track the 33 keypoints throughout the video and store their positions and visibilities. 
3. Kinematics Computations 
    > computes the kinematics of relevant keypoints (left and right wrist) as well as the Joint Angular Kinematics of joints of interest using the world coordinates generated from landmark extraction 
4. Compute Boxing Metrics 
    > determines the wrist velocity and shoulder and hip rotations from the computed kinematics in the previous stage
5. Adding Force Arrows 
    > (optional stage) Adds FBD to the video which shows where the force in the punch is being generated 
6. Fusion of Boxing Metrics to Video
    > outputs the input video with additional graphs to the right of it which track the boxing metrics and the estimated kniematics from pose detection