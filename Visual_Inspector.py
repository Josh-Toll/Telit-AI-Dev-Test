from ultralytics import YOLO
import cv2
import json
from pathlib import Path
import random
from datetime import datetime, timedelta
import numpy as np

#path configurations
INPUT_DIR = Path("images/")
OUTPUT_DIR = Path("outputs/")
CROPS_DIR = OUTPUT_DIR / "crops"


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)

LOCATIONS = ["zoo", "street", "restaurant", "river", "beach"]

if __name__ == "__main__":
    #load model
    model = YOLO("yolov8n-seg.pt")
    results = []

    threshold = float(input("Set minimum confidence threshold: "))

    #allow for user input to be either a percentage or a decimal
    if threshold > 1.0:
        threshold = threshold / 100.0

    classes_input = input("Enter target classes separated by commas (or leave blank for all): ")
    target_classes = [c.strip() for c in classes_input.split(",")] if classes_input != "" else []


    for idx, img_path in enumerate(INPUT_DIR.glob("*.*")):
        #yolo model inference
        outputs = model.predict(
            source=str(img_path),
            conf=float(threshold),
            save=True,
            save_txt=False,
            project=str(OUTPUT_DIR),
            name="annotated",
            classes=[k for k, v in model.names.items() if v in target_classes] if target_classes else None
        )
        
        img = cv2.imread(str(img_path))


        #random timestamp within the last 7 days (for the agentic section)
        now = datetime.now()            
        random_seconds = random.randint(0, 7 * 24 * 60 * 60)
        random_timestamp = now - timedelta(seconds=random_seconds)
        
        for output in outputs:
            if output.masks is not None:
                masks = output.masks.data.cpu().numpy()
            else:
                masks = []


            for i, box in enumerate(output.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                
                #save crop
                crop_dir = CROPS_DIR / cls_name
                crop_dir.mkdir(parents=True, exist_ok=True)
                crop_img = img[y1:y2, x1:x2]
                crop_fname = f"{img_path.stem}_{cls_name}_{x1}_{y1}_{x2}_{y2}.jpg"
                cv2.imwrite(str(crop_dir / crop_fname), crop_img)


                if len(masks) > i:
                    mask = (masks[i] * 255).astype(np.uint8)  # convert to 0-255
                    mask_path = OUTPUT_DIR / f"{img_path.stem}_{cls_name}_mask.png"
                    cv2.imwrite(str(mask_path), mask)


                
                results.append({
                    "image": img_path.name,
                    "class": cls_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],

                    #fake metadata for agentic section
                    "camera": f"CAM{idx}", 
                    "timestamp": str(random_timestamp),
                    "location": LOCATIONS[idx],
                })


    with open(OUTPUT_DIR / "detections.json", "w") as f:
        json.dump(results, f, indent=2)

