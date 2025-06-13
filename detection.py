from ultralytics import YOLO
from detectors.detector_local import emotions_map
import cv2

class YoloDetection:
    def __init__(self, model_name = "em_det_yolo_v1.pt"):
        # Download weights from https://github.com/ultralytics/ultralytics and change the path
        self.model = YOLO(f"./models/{model_name}")
        self.classNames = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

    def predict_and_detect(self, img, conf=0.5, rectangle_thickness=2, text_thickness=2, skip_classes=["disgust"]):
        results = self.model.predict(img, conf=conf)
        print(len(results))
        for result in results:
            for box in result.boxes:
                class_name = self.classNames[int(box.cls[0])]
                if class_name in skip_classes:
                    continue
                print({"class": list(map(lambda x: self.classNames[int(x)], box.cls)), "conf": box.conf})
                color = emotions_map.get(self.classNames[int(box.cls[0])])
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                              (int(box.xyxy[0][2]), int(box.xyxy[0][3])), color, rectangle_thickness)
                cv2.putText(img, f"{self.classNames[int(box.cls[0])]}".title(),
                            (int(box.xyxy[0][0])+5, int(box.xyxy[0][1]) + 20),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, color, text_thickness)
        return img, results

    def detect_from_image(self, image, conf = 0.3, skip_classes=['disgust']):
        result_img, _ = self.predict_and_detect(image, conf=conf, skip_classes=skip_classes)
        return result_img

def gen_frames(detector: YoloDetection):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (512, 512))
        if frame is None:
            break
        frame = detector.detect_from_image(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')