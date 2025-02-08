import os
import requests
from ultralytics import YOLO

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'license_plate_detector.pt')
MODEL_URL = 'https://raw.githubusercontent.com/mahdihuseine/object.detector/main/object/detector/models/license_plate_detector.pt'

class PlateRecognition:
    
    def detect_plate(self, image):
        try:

            if not os.path.exists(MODELS_DIR):
                os.makedirs(MODELS_DIR, exist_ok=True)

            if not os.path.exists(MODEL_PATH):
                try:
                    response = requests.get(MODEL_URL)
                    if response.status_code == 200:
                        with open(MODEL_PATH, 'wb') as f:
                            f.write(response.content)
                    else:
                        raise Exception("خطا در دانلود مدل!")
                except:
                    pass  # اگر خطایی در دانلو  د مدل بود، هیچ کاری انجام نمی‌دهیم

            self.license_plate_detector = YOLO(MODEL_PATH)

            if image is None:
                raise ValueError("تصویر وارد شده نامعتبر است!")

            # تشخیص پلاک با YOLO
            results = self.license_plate_detector(image)
            detections = results[0].boxes.data.tolist()

            if len(detections) == 0:
                raise ValueError("هیچ پلاکی شناسایی نشد!")

            # انتخاب بهترین تشخیص (با بیشترین امتیاز)
            detections = sorted(detections, key=lambda d: d[4], reverse=True)
            best_detection = detections[0]
            x1, y1, x2, y2, score, class_id = best_detection
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            return (x1, y1, x2, y2)

        except Exception as e:
            pass  # اگر خطایی در تشخیص پلاک وجود داشت، هیچ کاری انجام نمی‌دهیم
