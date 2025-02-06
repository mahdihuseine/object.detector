# object/detector/util.py
def get_car(plate_bbox, vehicle_detections):
    # تابع برای یافتن خودروهای مربوط به پلاک
    for vehicle in vehicle_detections:
        x1, y1, x2, y2, score = vehicle
        # بررسی اینکه آیا پلاک در داخل خودرو قرار دارد
        if x1 >= plate_bbox[0] and y1 >= plate_bbox[1] and x2 <= plate_bbox[2] and y2 <= plate_bbox[3]:
            return vehicle  # بازگشت مختصات خودرو
    return None  # اگر خودرو پیدا نشد
