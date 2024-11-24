import cv2
from ultralytics import YOLO

# Load your YOLO model
model = YOLO('../model/augmented/best.pt') 

label_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 
    25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 
    33: '7', 34: '8', 35: '9'
}
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    results = model.predict(frame)

    for result in results:
        boxes = result.boxes.xyxy.tolist()
        classes = result.boxes.cls.tolist()
        confidences = result.boxes.conf.tolist()

        for i in range(len(boxes)):
            if(confidences[i] > 0.5):
                x1, y1, x2, y2 = boxes[i]
                original_image = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  
                label = f"Conf: {confidences[i]:.2f} Class: {label_dict[int(classes[i])]}"
                original_image = cv2.putText(original_image, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
