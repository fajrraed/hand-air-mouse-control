import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import numpy as np
import math
import time

# 1. إعداد مكتبة MediaPipe لتتبع يد واحدة
model_path = 'hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(1)

# إعدادات الأداء لـ PyAutoGUI
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# --- متغيرات التحكم الذكي بالوضعية والتنعيم ---
scroll_enabled = False  # حفظ حالة السكرول (مفعل أم لا)
prev_y = None
prev_screen_x, prev_screen_y = 0, 0
last_click_time = 0

SMOOTHING = 0.25
FRAME_REDUCTION = 80
CLICK_COOLDOWN = 0.3

# أرقام مفاصل أطراف الأصابع الأربعة والأجزاء الموصلة لها (بدون الإبهام)
FINGER_TIPS = [8, 12, 16, 20]  # أطراف الأصابع: السبابة، الوسطى، البنصر، الخنصر
FINGER_PIPS = [6, 10, 14, 18]  # المفاصل الداخلية المقابلة لها

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # رسم إطار الحركة الأبيض (صندوق التحكم)
    cv2.rectangle(frame, (FRAME_REDUCTION, FRAME_REDUCTION),
                  (w - FRAME_REDUCTION, h - FRAME_REDUCTION), (255, 255, 255), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    frame_timestamp_ms = int(time.time() * 1000)
    results = detector.detect_for_video(mp_image, frame_timestamp_ms)

    if results.hand_landmarks:
        # تصحيح الخطأ: استخراج مصفوفة اليد الأولى المكتشفة مباشرة لتجنب خطأ الـ 'list'
        hand = results.hand_landmarks[0]

        # --- خوارزمية عد الأصابع لمعرفة حركة (خمسة) أو (حجرة) ---
        up_fingers = 0

        # 1. فحص الإبهام (مقارنة أفقية لطرف الإبهام مفصل 4 مع مفصل 3)
        if hand[4].x > hand[3].x:
            up_fingers += 1

        # 2. فحص بقية الأصابع الأربعة (تكون مرفوعة إذا كان الطرف أعلى عمودياً من المفصل الأوسط)
        for i in range(4):
            if hand[FINGER_TIPS[i]].y < hand[FINGER_PIPS[i]].y:
                up_fingers += 1

        # --- فحص الإيماءات وتغيير حالة السكرول ---
        if up_fingers == 5:
            scroll_enabled = True  # تشغيل وضع السكرول عند فتح اليد بالكامل "خمسة" 🖐️
        elif up_fingers == 0:
            scroll_enabled = False  # إيقاف وضع السكرول والعودة للماوس عند إغلاق اليد "حجرة" ✊

        # جلب نقاط التحكم الفردية (السبابة، الإبهام، الخنصر)
        thumb = hand[4]
        index_finger = hand[8]
        pinky_finger = hand[20]

        # ==========================================
        # 1. وضعية السكرول النشطة المستمرة
        # ==========================================
        if scroll_enabled:
            current_y = int(index_finger.y * h)
            if prev_y is not None:
                dy = prev_y - current_y
                if abs(dy) > 2:
                    pyautogui.scroll(dy * 6)  # تحريك الصفحة (اضرب في رقم أكبر لتسريع السكرول)
            prev_y = current_y

            # مؤشرات بصرية لوضع السكرول باللون الأزرق
            cv2.putText(frame, "SCROLL ACTIVE (Fist to Stop)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.circle(frame, (int(index_finger.x * w), int(index_finger.y * h)), 15, (255, 0, 0), cv2.FILLED)

        # ==========================================
        # 2. وضعية الماوس العادي والضغط (كليك)
        # ==========================================
        else:
            prev_y = None  # تصفير ذاكرة السكرول

            raw_x = int(index_finger.x * w)
            raw_y = int(index_finger.y * h)

            # ضبط الإحداثيات بالتناسب مع صندوق التحكم الداخلي لمنع التمدد الزائد لليد
            target_x = np.interp(raw_x, [FRAME_REDUCTION, w - FRAME_REDUCTION], [0, screen_w])
            target_y = np.interp(raw_y, [FRAME_REDUCTION, h - FRAME_REDUCTION], [0, screen_h])

            # تطبيق فلتر التنعيم للتخلص من اهتزاز مؤشر الماوس
            smoothed_x = prev_screen_x + (target_x - prev_screen_x) * SMOOTHING
            smoothed_y = prev_screen_y + (target_y - prev_screen_y) * SMOOTHING

            pyautogui.moveTo(int(smoothed_x), int(smoothed_y))
            prev_screen_x, prev_screen_y = smoothed_x, smoothed_y

            # حساب مسافات الضغط (تقريب الأصابع)
            left_click_dist = math.hypot(thumb.x - index_finger.x, thumb.y - index_finger.y)
            right_click_dist = math.hypot(thumb.x - pinky_finger.x, thumb.y - pinky_finger.y)

            current_time = time.time()
            if current_time - last_click_time > CLICK_COOLDOWN:
                # كليك يسار (تقريب الإبهام والسبابة)
                if left_click_dist < 0.03:
                    pyautogui.click(button='left')
                    last_click_time = current_time
                    cv2.circle(frame, (raw_x, raw_y), 15, (0, 255, 255), cv2.FILLED)  # ومضة صفراء

                # كليك يمين (تقريب الإبهام والخنصر)
                elif right_click_dist < 0.04:
                    pyautogui.click(button='right')
                    last_click_time = current_time
                    cv2.circle(frame, (raw_x, raw_y), 15, (0, 0, 255), cv2.FILLED)  # ومضة حمراء

                # حركة ماوس عادية بدون ضغط
                else:
                    cv2.putText(frame, "MOUSE MODE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.circle(frame, (raw_x, raw_y), 10, (0, 255, 0), cv2.FILLED)  # نقطة خضراء
    else:
        prev_y = None  # تصفير الحالة في حال خروج اليد من الكاميرا

    cv2.imshow("Hand Air Mouse - Smart Scroll Lock", frame)
    if cv2.waitKey(1) == 27:  # اضغط على زر ESC لإغلاق البرنامج بأمان
        break

cap.release()
cv2.destroyAllWindows()
