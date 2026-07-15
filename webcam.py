import os
import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# 설정

## 카메라 인덱스 (Photo Booth/스캔으로 확인한 번호. 0번 확인됨)
CAMERA_INDEX = 0

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
checkpoint = "epoch100.onnx"

# COCO 기준 person 클래스 번호
PERSON_CLASS_ID = 0


def main():
    # 모델 로드
    weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    model = YOLO(checkpoint)

    # 웹캠 열기 (파일이 아니라 인덱스라서 존재 여부 체크는 불필요)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: 인덱스 {CAMERA_INDEX}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 웹캠은 총 프레임 수 개념이 없어서(-1 또는 0으로 나옴) 별도 처리
    total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = total_frames_raw if total_frames_raw > 0 else None

    # 결과 저장 영상 준비
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    print(f"입력: 웹캠 인덱스 {CAMERA_INDEX} ({frame_w}x{frame_h}, {fps:.1f}fps)")
    print(f"결과 저장 위치: {output_path}")
    print("처리 중... 미리보기 창에서 'q'를 누르면 종료됩니다.")

    frame_idx = 0
    max_person_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽지 못했습니다. 카메라 연결을 확인하세요.")
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
            if total_frames:
                print(f"  {frame_idx}/{total_frames}")
            else:
                print(f"  {frame_idx}프레임 처리됨 (실시간)")

    cap.release()
    writer.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    print(f"완료. 최대 인원수: {max_person_count}명")
    print(f"결과 영상 저장됨: {output_path}")


if __name__ == "__main__":
    main()