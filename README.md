# ATV Deep Learning engineer test #
Version 2.0

### Task ###

Technical interview for AutomaticTV, senior Deep Learning engineer position.

The proposed coding interview is settled as a typical object detection problem using a subset of a custom dataset.
The data is organized as a frame - XML pair. To receive the data, please ask the interviewing committee.

The first part consists on building the dataset reader function from the XML files, the second part consists on 
building the training loop of the object detector. 



### How do I get set up? ###
This test is built upon python 3.6, take that into account when building the virtual environment.

    pip install -r requirements.txt

If needed, feel free to use any other packages.

Create data and models directories.

    mkdir data models

Unzip the provided file in the **data** folder.

_(Although the test is thought to be in Python, feel free to use any other language you like better.)_



### Main task description: ###
Given the video and annotated dataset:
1. Complete the dataset reader function **parse_info_from_xml** in **dataset.py**
2. Build and **train an object detector** module that can successfully detect the players and ball in a video.  
    _Hint_: You can base your solution on https://colab.research.google.com/drive/1_GdoqCJWXsChrOiY8sZMr_zbr_fH-0Fg and use the existing collab interface to train directly on the cloud.
3. Use the provided video to **test your constructed model**.  
    Metrics on how well it performs on ground truth are well appreciated.
4. _(Optional implementation)_ Try to implement a **team detector** to know which team each player belongs to.
At least, try to think how you would face this problem and which steps would you take.



### Dataset structure ###
Inside folder **data**:

    |-- match1
        |-- file1.jpg
        |-- file1.xml
        ...
        |-- fileN.jpg
        |-- fileN.xml
       
    |-- matchK
        |-- file1.jpg
        |-- file1.xml
        ...
        |-- fileN.jpg
        |-- fileN.xml
    

        
## Aspects to evaluate
### Language and libraries 
* Python
* Pytorch
* OpenCV

### Deliverables
* Source code
* Results
* Optional: a file with observed things and tensorboard results (graphs, loss functions, ...)

### What we value
* Coding structure and habits
* Solvency given a problem

### Recommendations:
* Use already imported packages
* Use documentation