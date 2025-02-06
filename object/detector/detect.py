# object/detector/detect.py
from ultralytics import YOLO
import cv2
import numpy as np
import os
import util  # فرض بر این است که توابع read_license_plate و get_car در این ماژول موجود است.

def detect_license_plate(image_path):
    coco_model = YOLO('yolov8n.pt')  # مدل تشخیص عمومی (خودروها و سایر اشیاء)
    license_plate_detector = YOLO(os.path.join(os.path.dirname(__file__), 'models', 'license_plate_detector.pt'))  # مدل تشخیص پلاک

    image = cv2.imread(image_path)

    if image is None:
        print("تصویر خوانده نشد. لطفاً آدرس تصویر را بررسی کنید.")
        return

    vehicles = [2, 3, 5, 7]  # کلاس‌های خودرو

    # تشخیص خودروها در تصویر
    results_vehicle = coco_model(image)[0]
    vehicle_detections = []

    for detection in results_vehicle.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            vehicle_detections.append([x1, y1, x2, y2, score])

    # تشخیص پلاک خودرو
    results_lp = license_plate_detector(image)[0]

    output_dir_plate = 'plates'
    output_dir_car = 'cars'
    os.makedirs(output_dir_plate, exist_ok=True)
    os.makedirs(output_dir_car, exist_ok=True)

    plate_count = 0
    for detection in results_lp.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        # رسم کادر پلاک و ذخیره تصاویر
        plate_crop = image[int(y1):int(y2), int(x1):int(x2)]
        plate_filename = os.path.join(output_dir_plate, f"plate_{plate_count}.jpg")
        cv2.imwrite(plate_filename, plate_crop)
        print(f"تصویر پلاک در {plate_filename} ذخیره شد.")

        # تطبیق پلاک با خودرو
        car_bbox = util.get_car([x1, y1, x2, y2], vehicle_detections)
        if car_bbox:
            car_x1, car_y1, car_x2, car_y2, car_score = car_bbox
            car_crop = image[int(car_y1):int(car_y2), int(car_x1):int(car_x2)]
            car_filename = os.path.join(output_dir_car, f"car_{plate_count}.jpg")
            cv2.imwrite(car_filename, car_crop)
            print(f"تصویر خودرو مربوط به پلاک در {car_filename} ذخیره شد.")
        plate_count += 1

    # نمایش تصویر
    cv2.imshow("Detected License Plates", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output_path = 'output.jpg'
    cv2.imwrite(output_path, image)
    print(f"تصویر نهایی در {output_path} ذخیره شد.")
