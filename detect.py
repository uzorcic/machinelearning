import cv2
import torch
import sys
from ultralytics import YOLO

# Load the pre-trained YOLOv8 model
#model = torch.hub.load('ultralytics/yolov8', 'yolov8x', pretrained=True, trust_repo=True)  # 'yolov8x' is a large model
#model = YOLO('runs/detect/train28/weights/best.pt') #100 epochs tube
model = YOLO('best.pt')



def detect_objects(frame, model):
    # Perform inference
    results = model(frame, verbose=False)
    #for result in results:
        #print("rezultat: -----------> " + str(result.boxes.xyxy[0]))

    #Process results
    labels, cords, confidences = [], [], []
    for det in results:
        for i in range(len(det.boxes.xyxy)):
            labels.append(det.boxes.cls[i])
            cords.append(det.boxes.xyxy[i])  # xyxy format
            confidences.append(det.boxes.conf[i])
        #print("conf: ---->" + str(det.boxes.conf))
        #print("size: ----->" + str(det.size))
    return labels, cords, confidences

    #print(results)

def draw_detections(frame, labels, cords, confidences, names, is_detectionframe):
    h, w, _ = frame.shape
    for label, bbox, confidence in zip(labels, cords, confidences):
        if confidence > 0.3:
            # Convert tensor to a list and then to individual coordinates
            #print("bbox: " + str(bbox.tolist()))
            #print("size: " + str(len(bbox.tolist())))
            bbox_list = bbox.tolist()
            #print(bbox_list)
            x1, y1, x2, y2 = bbox_list
            x1, y1, x2, y2 = int(x1 * 1), int(y1 * 1), int(x2 * 1), int(y2 * 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{names[int(label)]} {confidence:.2f}"
            if is_detectionframe and names[int(label)]!="person":
                print(label_text)
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



def play_webcam():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        sys.exit()

    frame_count = 0
    old_labels, old_cord, old_confidences, old_modelnames = [],[],[],[]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read video frame")
            break

        frame_count += 1
        if frame_count % 50 == 0:
            labels, cord, confidences = detect_objects(frame, model)
            draw_detections(frame, labels, cord, confidences, model.names, True)
            old_labels, old_cord, old_confidences, old_modelnames = labels, cord, confidences, model.names
        else:
            if len(old_labels) > 0:
                draw_detections(frame, old_labels, old_cord, old_confidences, old_modelnames, False)

        cv2.imshow('HLS Stream with YOLOv8 Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #stream_url = "http://<streamingserver>/index.m3u8"
    play_webcam()
