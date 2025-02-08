import cv2
from ObjectRecognition import PlateRecognition

def main():
    video_path = r"C:\Users\Administrator\Desktop\mstf\video2.mp4"
    
    # باز کردن ویدئو
    cap = cv2.VideoCapture(video_path)

    # بررسی باز شدن ویدئو
    if not cap.isOpened():
        print("خطا در باز کردن فایل ویدئو")
        return

    # ایجاد شیء از کلاس PlateRecognition
    plate_recognition = PlateRecognition()

    while True:
        # خواندن هر فریم از ویدئو
        ret, frame = cap.read()
        if not ret:
            break

        # شناسایی پلاک و دریافت مختصات
        plate_coords = plate_recognition.detect_plate(frame)
        
        # بررسی معتبر بودن مختصات
        if plate_coords and len(plate_coords) == 4:
            x1, y1, x2, y2 = plate_coords
            if 0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]:
                # رسم کادر روی تصویر
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                print("مختصات نامعتبر پلاک")
        else:
            print("خطا در دریافت مختصات پلاک")

        # نمایش تصویر
        cv2.imshow('Video Frame with Plate Detection', frame)

        # توقف با فشردن کلید 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # آزاد کردن منابع
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
