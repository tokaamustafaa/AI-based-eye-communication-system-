"""
camera_setup.py — iVCam USB helper for the gaze system.
"""

import cv2


def open_camera(prefer_droidcam: bool = True,
                width: int = 640,
                height: int = 480) -> cv2.VideoCapture:
    
    for idx in [1, 0]:
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ret, frame = cap.read()
        if ret and frame is not None:
            # تأكد إن الصورة مش خضراء
            b, g, r = cv2.split(frame)
            is_green = g.mean() > 150 and b.mean() < 50 and r.mean() < 50
            if not is_green:
                print(f"[Camera] ✓ Camera connected at index {idx}")
                return cap
        cap.release()

    raise RuntimeError("[Camera] No working camera found!")


def set_droidcam_ip(ip: str, port: int = 4747):
    pass


# ---------------------------------------------------------------------------
# Quick test  (python camera_setup.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cap = open_camera()
    print("Press Q to quit the preview.")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[Camera] No frame received.")
            break
        cv2.imshow("Camera Test — press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
