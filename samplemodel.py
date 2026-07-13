import os
import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# 설정

## 입력 영상 경로
INPUT_VIDEO_PATH = "/Users/choejihu/Documents/G-code/PSnn/sampledata/2/4791196-hd_1920_1080_30fps.mp4"
## 결과 저장 폴더 경로
OUTPUT_DIR = "result"                       

## 결과 파일명           
OUTPUT_FILENAME = "result1.avi" 

### 처리 중 화면 미리보기 여부 (창 띄우기 싫으면 False)
SHOW_WINDOW = True    
### 확신도
CONF_THRESHOLD = 0.4  

# 모델
REPO_ID = "Ultralytics/YOLO26"
FILENAME = "yolo26x.pt"
checkpoint = "best.pt"

# COCO 기준 person 클래스 번호
PERSON_CLASS_ID = 0


def main():
    # 모델 로드
    weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    model = YOLO(checkpoint)

    # 입력 영상 열기
    if not os.path.exists(INPUT_VIDEO_PATH):
        raise FileNotFoundError(f"입력 영상을 찾을 수 없습니다: {INPUT_VIDEO_PATH}")

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {INPUT_VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 결과 저장 영상 준비
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    print(f"입력 영상: {INPUT_VIDEO_PATH} ({frame_w}x{frame_h}, {fps:.1f}fps, 총 {total_frames}프레임)")
    print(f"결과 저장 위치: {output_path}")
    print("처리 중... 중간에 멈추려면 미리보기 창에서 'q'를 누르세요.")

    frame_idx = 0
    max_person_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # person 클래스만 탐지
        results = model.predict(frame, classes=[PERSON_CLASS_ID], conf=CONF_THRESHOLD, verbose=False)
        boxes = results[0].boxes
        person_count = len(boxes) if boxes is not None else 0
        max_person_count = max(max_person_count, person_count)

        if boxes is not None:
            for box in boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"People: {person_count}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )

        writer.write(frame)

        if SHOW_WINDOW:
            cv2.imshow("Person Count (processing)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("사용자에 의해 중단되었습니다.")
                break

        if frame_idx % 30 == 0:
            print(f"  {frame_idx}/{total_frames}")

    cap.release()
    writer.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    print(f"완료. 최대 인원수: {max_person_count}명")
    print(f"결과 영상 저장됨: {output_path}")


if __name__ == "__main__":
    main()