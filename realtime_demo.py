import os
import cv2
import torch
import numpy as np
from collections import deque
from torchvision import transforms
from models.multitask_model import MultiTaskFaceModel


def open_camera():
    # Windows-friendly webcam open
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
    return None


def clamp_int(x, lo=0, hi=100):
    return max(lo, min(int(x), hi))


def softmax_np(logits_1x2: torch.Tensor):
    """
    logits_1x2: torch tensor shape [1, 2]
    returns: np array shape [2] with probabilities
    """
    probs = torch.softmax(logits_1x2, dim=1).detach().cpu().numpy()[0]
    return probs


def face_change_score(face_a_rgb, face_b_rgb, size=(64, 64)):
    """
    Returns a score in [0, 1] where:
    0 = very similar face crops
    1 = very different face crops
    """
    a = cv2.resize(face_a_rgb, size, interpolation=cv2.INTER_AREA)
    b = cv2.resize(face_b_rgb, size, interpolation=cv2.INTER_AREA)

    # Normalize to reduce lighting effects
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)

    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)

    mad = np.mean(np.abs(a - b))
    score = float(np.clip(mad / 2.5, 0.0, 1.0))
    return score


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ✅ Use spoof-finetuned weights (IMPORTANT)
    # If you don't have this file yet, run: python finetune_spoof.py
    weights_path = "multitask_model_spoof.pth"

    if not os.path.exists(weights_path):
        print(f"❌ Model weights not found: {weights_path}")
        print("If you have only multitask_model.pth, spoof will always look REAL.")
        print("Run: python finetune_spoof.py  (it will create multitask_model_spoof.pth)")
        return

    model = MultiTaskFaceModel().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # ---- Face detector ----
    cascade_path = "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        print(f"❌ Haarcascade not found: {cascade_path}")
        return
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # ---- Preprocessing ----
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ---- Webcam ----
    cap = open_camera()
    if cap is None:
        print("❌ Could not open webcam. Close Zoom/Teams/browser camera tabs and check permissions.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("✅ Webcam opened.")
    print("Controls: Q/Esc quit | R reset | L toggle age lock")

    # ---- Settings ----
    AGE_WINDOW = 30
    LOCK_AFTER_SECONDS = 3.0
    FACE_CHANGE_THRESHOLD = 0.28
    NO_FACE_RESET_FRAMES = 15

    # ✅ Spoof smoothing settings (PRO)
    SPOOF_WINDOW = 15  # store last 15 spoof probabilities
    spoof_prob_buffer = deque(maxlen=SPOOF_WINDOW)
    SPOOF_DECISION_THRESHOLD = 0.60  # if avg prob_real >= 0.60 => REAL else FAKE

    age_buffer = deque(maxlen=AGE_WINDOW)
    locked_age = None
    lock_enabled = True
    start_ticks = cv2.getTickCount()

    prev_face_rgb = None
    no_face_count = 0

    def reset_all():
        nonlocal locked_age, start_ticks, prev_face_rgb, no_face_count
        age_buffer.clear()
        spoof_prob_buffer.clear()
        locked_age = None
        start_ticks = cv2.getTickCount()
        prev_face_rgb = None
        no_face_count = 0

    cv2.namedWindow("Face Analysis", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        if len(faces) == 0:
            no_face_count += 1
            if no_face_count >= NO_FACE_RESET_FRAMES:
                reset_all()
            cv2.imshow("Face Analysis", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            if key in [ord('r'), ord('R')]:
                reset_all()
            if key in [ord('l'), ord('L')]:
                lock_enabled = not lock_enabled
                if not lock_enabled:
                    locked_age = None
            continue

        no_face_count = 0

        # Use largest face
        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        (x, y, w, h) = faces[0]

        face = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # If person changes, reset lock + smoothing buffers
        if prev_face_rgb is not None:
            change = face_change_score(prev_face_rgb, face_rgb)
            if change > FACE_CHANGE_THRESHOLD:
                reset_all()
        prev_face_rgb = face_rgb

        # ---- Inference ----
        face_tensor = transform(face_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_age, pred_gender, pred_spoof = model(face_tensor)

        # ---- Age smoothing ----
        raw_age = float(pred_age.item())
        age_buffer.append(raw_age)
        smoothed_age = sum(age_buffer) / len(age_buffer)
        smoothed_age = clamp_int(round(smoothed_age), 0, 100)

        # ---- Lock age (optional) ----
        if lock_enabled and locked_age is None:
            elapsed = (cv2.getTickCount() - start_ticks) / cv2.getTickFrequency()
            if elapsed >= LOCK_AFTER_SECONDS and len(age_buffer) == AGE_WINDOW:
                locked_age = smoothed_age

        display_age = locked_age if (lock_enabled and locked_age is not None) else smoothed_age

        # ---- Gender ----
        # UTKFace commonly uses: 0=male, 1=female
        gender_idx = torch.argmax(pred_gender, dim=1).item()
        gender = "Male" if gender_idx == 0 else "Female"

        # ---- Spoof with probability + smoothing ----
        spoof_probs = softmax_np(pred_spoof)  # [p_fake, p_real] OR [p0, p1] depending on training
        # IMPORTANT: In our fine-tune we used label 1=REAL, 0=FAKE.
        # CrossEntropy classes are: 0=FAKE, 1=REAL -> so index 1 is REAL probability.
        prob_real = float(spoof_probs[1])
        spoof_prob_buffer.append(prob_real)

        avg_prob_real = float(sum(spoof_prob_buffer) / len(spoof_prob_buffer))
        spoof = "Real" if avg_prob_real >= SPOOF_DECISION_THRESHOLD else "Fake"

        # Confidence to show on screen
        conf = avg_prob_real if spoof == "Real" else (1.0 - avg_prob_real)
        conf_pct = int(round(conf * 100))

        color = (0, 255, 0) if spoof == "Real" else (0, 0, 255)

        # ---- Draw UI ----
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        lock_txt = "LOCKED" if (lock_enabled and locked_age is not None) else "SMOOTH"
        label1 = f"Age: {display_age} ({lock_txt}) | Gender: {gender}"
        label2 = f"Spoof: {spoof}  Conf: {conf_pct}%  (avgReal={avg_prob_real:.2f})"

        cv2.putText(frame, label1, (x, max(30, y - 28)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, label2, (x, max(55, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Face Analysis", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:
            break
        if key in [ord('r'), ord('R')]:
            reset_all()
        if key in [ord('l'), ord('L')]:
            lock_enabled = not lock_enabled
            if not lock_enabled:
                locked_age = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
