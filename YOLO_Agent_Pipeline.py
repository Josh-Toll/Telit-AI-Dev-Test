import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re
from ultralytics import YOLO
import cv2
import json
from datetime import datetime, timedelta
import random
from collections import Counter


VIDEO_PATH = "20250815_131659.mp4" 
OUTPUT_JSON = "video_detections.json"
CONF_THRES = 0.3
MODEL_PATH = "yolov8n-seg.pt"
CLASSES = ["elephant", "person", "car", "horse", "bottle", "cup", "chair", "potted plant"] #these classes can be found in the dataset
LOCATIONS = ["zoo", "street", "restaurant", "river", "beach", "outside"]


#run YOLO on video and save detections to JSON
def detect_objects():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_idx = 0
    results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = model.predict(source=frame, conf=CONF_THRES, verbose=False)

        for o in output:
            for i, box in enumerate(o.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])


                mask_points = None
                if o.masks is not None:
                    mask_points = o.masks.xy[i].tolist()  # polygon points

                #fake metadata
                start = datetime.now()
                curr_time = start + timedelta(seconds=frame_idx / 60) #assuming 60 FPS video


                results.append({
                    "frame": frame_idx,
                    "class": cls_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "timestamp": str(curr_time),
                    "camera": 5,
                    "location": "outside"
                })

        frame_idx += 1

    cap.release()

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Saved detections to {OUTPUT_JSON}")


def add_embeddings(documents, embed_model):
    for doc in documents:
        doc["embedding"] = embed_model.encode(doc["text"], convert_to_numpy=True)

def build_documents(detections, embed_model):
    docs = []
    for d in detections:
        text = f"Camera {d['camera']} detected a {d['class']} with confidence {d['confidence']:.2f} at {d['timestamp']} in {d['location']}."
        docs.append({
            "text": text,
            "metadata": d
        })
    add_embeddings(docs, embed_model)
    return docs

def query_rag(question, documents, embed_model, k=5):
    #extract filters
    date, camera, location, obj_class = None, None, None, None

    #date
    for word in question.split():
        try:
            date = datetime.strptime(word, "%Y-%m-%d").date()
            break
        except ValueError:
            continue

    #camera
    cam_match = re.search(r'CAM(\d+)', question, re.IGNORECASE)
    if cam_match:
        camera = int(cam_match.group(1))

    #location
    for loc in LOCATIONS:
        if loc.lower() in question.lower():
            location = loc
            break

    #object class
    for c in CLASSES:
        if c.lower() in question.lower():
            obj_class = c
            break

    #filter documents
    filtered_docs = documents
    if date:
        filtered_docs = [d for d in filtered_docs
                         if datetime.fromisoformat(d["metadata"]["timestamp"]).date() == date]
    if camera:
        filtered_docs = [d for d in filtered_docs if d["metadata"]["camera"] == camera]
    if location:
        filtered_docs = [d for d in filtered_docs if d["metadata"]["location"].lower() == location.lower()]
    if obj_class:
        filtered_docs = [d for d in filtered_docs if d["metadata"]["class"].lower() == obj_class.lower()]

    if not filtered_docs:
        return "No detections found for your query."

    #FAISS search
    emb_matrix = np.stack([d["embedding"] for d in filtered_docs])
    q_emb = embed_model.encode(question, convert_to_numpy=True)
    index = faiss.IndexFlatL2(emb_matrix.shape[1])
    index.add(emb_matrix)
    distances, indices = index.search(np.array([q_emb]), k)

    retrieved_docs = [filtered_docs[i]["text"] for i in indices[0] if i < len(filtered_docs)]
    return retrieved_docs[0]  #return top result


if __name__ == "__main__":
    detect_objects()

    with open(OUTPUT_JSON) as f:
        detections = json.load(f)

    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    documents = build_documents(detections, embed_model)

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", "stop"]:
            break
        response = query_rag(user_input, documents, embed_model)
        print(f"Response: {response}")