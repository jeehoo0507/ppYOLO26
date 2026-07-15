import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        continue
    ret, frame = cap.read()
    if ret:
        cv2.putText(frame, f"Index {i} (any key = next)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow(f"Camera {i}", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cap.release()