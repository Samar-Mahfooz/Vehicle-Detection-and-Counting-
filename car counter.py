import cv2
import cvzone
import math

from ultralytics import YOLO
from sort import *

classNames = {
 0: 'person',
 1: 'bicycle',
 2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 6: 'train',
 7: 'truck',
 8: 'boat',
 9: 'traffic light',
 10: 'fire hydrant',
 11: 'stop sign',
 12: 'parking meter',
 13: 'bench',
 14: 'bird',
 15: 'cat',
 16: 'dog',
 17: 'horse',
 18: 'sheep',
 19: 'cow',
 20: 'elephant',
 21: 'bear',
 22: 'zebra',
 23: 'giraffe',
 24: 'backpack',
 25: 'umbrella',
 26: 'handbag',
 27: 'tie',
 28: 'suitcase',
 29: 'frisbee',
 30: 'skis',
 31: 'snowboard',
 32: 'sports ball',
 33: 'kite',
 34: 'baseball bat',
 35: 'baseball glove',
 36: 'skateboard',
 37: 'surfboard',
 38: 'tennis racket',
 39: 'bottle',
 40: 'wine glass',
 41: 'cup',
 42: 'fork',
 43: 'knife',
 44: 'spoon',
 45: 'bowl',
 46: 'banana',
 47: 'apple',
 48: 'sandwich',
 49: 'orange',
 50: 'broccoli',
 51: 'carrot',
 52: 'hot dog',
 53: 'pizza',
 54: 'donut',
 55: 'cake',
 56: 'chair',
 57: 'couch',
 58: 'potted plant',
 59: 'bed',
 60: 'dining table',
 61: 'toilet',
 62: 'tv',
 63: 'laptop',
 64: 'mouse',
 65: 'remote',
 66: 'keyboard',
 67: 'cell phone',
 68: 'microwave',
 69: 'oven',
 70: 'toaster',
 71: 'sink',
 72: 'refrigerator',
 73: 'book',
 74: 'clock',
 75: 'vase',
 76: 'scissors',
 77: 'teddy bear',
 78: 'hair drier',
 79: 'toothbrush'
 }

cap = cv2.VideoCapture("C:\\Users\\Samar\\Downloads\\cars.mp4")
mask = cv2.imread("C:\\Users\\Samar\\Downloads\\mask1.png")
cap.set(3,1280)
cap.set(4,720)
model = YOLO('../Yolo Weights/yolov8l.pt')
tracker = Sort(max_age=20)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
limits = [125,420,873,420]
# new_height = 800
# new_width = 450
print(width,height)


class_required = ["car","motorcycle","bus","truck"]
total_count =[]
while True:
    success, img = cap.read()
    important_region = cv2.bitwise_and(img,mask)
    # img = cv2.resize(img, (new_width,new_height))
    results = model(important_region, stream=True)
    detections = np.empty((0,5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            w ,h = x2-x1 , y2-y1
            confidence_value = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            class_name = classNames[cls]
            if(confidence_value>0.2) and class_name in class_required:
                #cvzone.cornerRect(img,(x1,y1,w,h))
                cvzone.putTextRect(img, f'{classNames[cls]} {confidence_value}', (max(0,x1),max(35,y1-20)), scale=1,thickness=2)
                currentArray = np.array([x1,y1,x2,y2,confidence_value])
                detections = np.vstack((detections,currentArray))

    resultTracker =tracker.update(detections)
    cv2.line(img,(limits[0],limits[1]), (limits[2],limits[3]), color=(255,0,255), thickness=2)
    for result1 in resultTracker:
        x1,y1,x2,y2,id = result1
        x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
        w,h = x2-x1,y2-y1
        cx,cy = x1  + w//2, y1 + h//2
        cvzone.cornerRect(img,(x1,y1,w,h), l=9,rt=3)
        cv2.circle(img,(cx,cy), 5,(255,0,255), cv2.FILLED)
        if limits[0]<cx<limits[2] and (limits[1]-15<cy<limits[1]+15):
            if not(id in total_count):
                total_count.append(id)
        cvzone.putTextRect(img, f'Count : {len(total_count)}', (50,50), scale=2, thickness=2,offset=10)
        cv2.line(img,(limits[0],limits[1]), (limits[2],limits[3]), color=(0,255,0), thickness=2)
    cv2.imshow("Image",img)
    cv2.waitKey(35)