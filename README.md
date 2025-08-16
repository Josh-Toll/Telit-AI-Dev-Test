# AI Developer Home Test

## Setup Instructions
 - Python 3.12.7
 - Clone this repository or download the files, keeping them in the same relative paths.
 - Install the required packages from the requirements.txt
     - Open the terminal.
     - Navigate to the directory that contains this project.
     - Create a new virtual environment (optional).
       - Type python -m venv venv into the terminal.
       - macOS/Linux: source venv/bin/activate   
       - Windows: venv\Scripts\activate
     - Install the dependencies: pip install -r requirements.txt
       - If numpy has dependency issues, downgrade to numpy version 1.26.4

## This project contains 3 different python files as well as a notebook file. 
  - The notebook file, "AI_Dev_Home_Test.ipynb" demonstrates how each part of the project works on one file.
  - "Visual_Inspector.py" is a YOLO object detection model that allows the user to input a minimum confidence level and choose which object classes to detect.
  - "Agent.py" is an agentic AI that allows the user to input a request for the AI to parse the data received from the YOLO object detection model.
  - YOLO_Agent_Pipeline.py" combines both of the previous parts and builds on them. The YOLO model in this file runs on every frame of a quick .mp4 video (around 11 seconds). The assistant allows users to input a natural language request, which the RAG-based model will use to parse the data from the short video and respond in a more natural way.


  
## How to Run
  - Open the terminal.
  - Navigate to the directory that contains this project.
  - Activate virtual environment (if applicable).
  - Select a file to run.
    - Type into the terminal "python Visual_Inspector.py" for the Visual Inspector.
    - Type into the terminal "python Agent.py" for the Agent.
    - Type into the terminal "python YOLO_Agent_Pipeline.py" for the YOLO Agent Pipeline.
  - To stop "Agent.py" or "YOLO_Agent_Pipeline.py", type "stop" or "exit" when prompted for an input.

  - To run "AI_Dev_Home_Test.ipynb", the file can be downloaded and run in jupyter-notebook or google colab.


## Considerations and Assumptions
  ### "Visual_Inspector.py"
  - Yolo model weights download from Ultralytics auto-downloads upon running the file for the first time.
  - Confidence level can be input as an integer (eg. 50 -> 50%) or as a decimal from 0-1 (eg. 0.5 -> 50%).
  - Target classes must be separated by a comma.
  - To search for all classes that YOLOv8 can comprehend, just press enter instead of inputting any target classes.
  - Outputs go to the output folder.
    - Output folder contains all of the annotated images, segmentation masks, and the detection JSON file.
    - Example output for detections.json:
    {
    "image": "000000356125.jpg",
    "class": "person",
    "confidence": 0.6741284132003784,
    "bbox": [
      0,
      105,
      68,
      237
    ],
    "camera": "CAM0",
    "timestamp": "2025-08-11 12:22:50.382078",
    "location": "zoo"
    }

  ### "Agent.py"
  - Agent.py can handle multiple tasks in a single request.
  - Any dates specified by the user must be in the format YYYY-MM-DD.
  - Camera names specified must be in the format "CAM1", "CAM2", etc.
  - User requests, request parameters, and assistant responses are documented in the requests.json file.
    - Example output for requests.json:
      {
    "user": {
      "request type": "summarize_events",
      "parameters": {
        "camera": null,
        "date": null,
        "location": "zoo"
      }
    },
    "Assistant": "Summary for {'camera': None, 'date': None, 'location': 'zoo'}: 3 notable detections."
  },

  
  ### "YOLO_Agent_Pipeline.py"
  - The RAG model used here does not realize that "people" is associated with "person", so questions about how many people must be phrased like "Which camera saw the most persons yesterday?"
  - Any dates specified by the user must be in the format YYYY-MM-DD.
  - Camera names specified must be in the format "CAM1", "CAM2", etc.
  - YOLO_Agent_Pipeline.py detects objects on every frame of the input video, so a frame count has been added to replace the image name.
  - Object detections are recorded in video_detections.json.
    - Example output for video_detections.json:
      {
    "frame": 0,
    "class": "car",
    "confidence": 0.9052159190177917,
    "bbox": [
      191,
      633,
      587,
      801
    ],
    "timestamp": "2025-08-15 17:27:54.202748",
    "camera": 5,
    "location": "outside"
  }

