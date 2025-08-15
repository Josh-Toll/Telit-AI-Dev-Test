from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import re
from pathlib import Path
import torch
from datetime import datetime


YOLO_JSON_PATH = Path("outputs/detections.json")
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
INTENT_LABELS = ["fetch_latest_detections", "summarize_events", "report_statistics"]


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

CAMERAS = [f"CAM{i}" for i in range(1, 6)]
LOCATIONS = ["zoo", "street", "restaurant", "river", "beach"]

#extract parameters from user request
def extract_parameters(text):
    params = {"camera": None, "date": None, "location": None}

    for cam in CAMERAS:
        if cam.lower() in text.lower():
            params["camera"] = cam

    for loc in LOCATIONS:
        if loc.lower() in text.lower():
            params["location"] = loc

    for word in text.split():
        try:
            datetime.strptime(word, "%Y-%m-%d")
            params["date"] = word
        except ValueError:
            pass

    return params

#classify intents into "fetch_latest_detections", "summarize_events", or "report_statistics"
def classify_request(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        intent_id = torch.argmax(logits, dim=-1).item()
        intent = INTENT_LABELS[intent_id % len(INTENT_LABELS)]
    return intent

#filter documents based on user request
def filter_documents(documents, params):
    filtered = documents
    if params.get("date"):
        filtered = [d for d in filtered if d["date"].startswith(params["date"])]
    if params.get("camera"):
        filtered = [d for d in filtered if str(d["camera"]) == params["camera"]]
    if params.get("location"):
        filtered = [d for d in filtered if params["location"].lower() in d["location"].lower()]
    return filtered


def fetch_latest_detections(params):
    if not YOLO_JSON_PATH.exists():
        return "No detection data found."

    with open(YOLO_JSON_PATH) as f:
        detections = json.load(f)

    filtered = filter_documents(detections, params)
    return filtered

def summarize_events(params):
    with open(YOLO_JSON_PATH) as f:
        detections = json.load(f)

    filtered = filter_documents(detections, params)    
    return f"Summary for {params or 'all data'}: {len(filtered)} notable detections."


def report_statistics(params):
    with open(YOLO_JSON_PATH) as f:
        detections = json.load(f)
        
    filtered = filter_documents(detections, params)

    total_detections = len(filtered)
    unique_classes = len(set(d["class"] for d in filtered))
    return f"Statistics for {params or 'all data'}: {total_detections} detections, {unique_classes} unique classes."


#handle request, then run the appropriate actions
def handle_request(user_input):

    #the embedding model has trouble classifying requests as "report_statistics", so it is handled separately here. 
    if any(word in user_input.lower() for word in ["stats", "statistics", "report"]):
        intent = "report_statistics"
    else:
        intent = classify_request(user_input)
    params = extract_parameters(user_input)
    

    #multi-part request handling
    if " and " in user_input.lower():
        parts = [p.strip() for p in user_input.split(" and ")]
        responses = []
        for p in parts:
            sub_intent = classify_request(p)
            sub_params = extract_parameters(p)
            responses.append(run_action(sub_intent, sub_params))
        return responses
    else:
        return run_action(intent, params)

#run the action based on the intent, return the appropriate result
def run_action(intent, params):
    try:
        with open("requests.json", "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []

    

    if intent == "fetch_latest_detections":
        response =  fetch_latest_detections(params)
    elif intent == "summarize_events":
        response = summarize_events(params)
    elif intent == "report_statistics":
        response =  report_statistics(params)
    else:
        response = "Sorry, I didn't understand your request."

    entry = {
        "user": {
            "request type": intent,
            "parameters": params,
        },
        "Assistant": response
    }

    history.append(entry)
    with open("requests.json", "w") as f:
        json.dump(history, f, indent=2)
    return response

if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", "stop"]:
            break
        response = handle_request(user_input)
        print(f"Assistant: {response}")