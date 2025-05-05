import cv2
import os
import time
import preprocessing_config as config
import numpy as np
from datetime import datetime


def extract_frames(video_path, output_base_folder, frame_interval, target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE)):
    movie_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(output_base_folder, movie_name)
    os.makedirs(output_folder, exist_ok=True)

    existing_files = [f for f in os.listdir(output_folder) if f.endswith('.jpg')]
    existing_numbers = sorted(
        [int(f.split("_")[-1].split(".")[0]) for f in existing_files if f.split("_")[-1].split(".")[0].isdigit()])
    next_frame_number = existing_numbers[-1] + 1 if existing_numbers else 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không thể mở phim: {video_path}")
        return

    start_time = time.time()
    start_sys_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            resized_frame = cv2.resize(frame, target_size)
            frame_filename = f"{movie_name}_{next_frame_number:05d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, resized_frame)
            next_frame_number += 1
        frame_count += 1

    cap.release()

    end_time = time.time()
    elapsed_time = end_time - start_time
    end_sys_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print(f"Đã trích xuất {next_frame_number} frames từ phim {video_path} vào thư mục {output_folder}.")
    print(f"Thời gian hoàn thành: {elapsed_time:.2f} giây ({start_sys_time} → {end_sys_time}).\n")


def process_all_videos(movies_folder=os.path.join(config.ROOT_DIR, "MOVIES"),
                       datasets_folder=os.path.join(config.ROOT_DIR, "PRE_DATASET"),
                       frame_interval=config.FRAME_INTERVAL):
    os.makedirs(datasets_folder, exist_ok=True)

    video_files = [f for f in os.listdir(movies_folder) if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))]

    if not video_files:
        print("Không tìm thấy phim nào trong thư mục. Vui lòng thêm phim và thử lại.")
        return

    start_time_all = time.time()
    start_sys_time_all = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for movie_file in video_files:
        video_path = os.path.join(movies_folder, movie_file)
        print(f"Đang trích xuất: {movie_file}")
        extract_frames(video_path, datasets_folder, frame_interval)

    end_time_all = time.time()
    total_time = end_time_all - start_time_all
    end_sys_time_all = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print(f"Hoàn thành toàn bộ phim lúc: {end_sys_time_all}")
    print(f"Tổng thời gian xử lý: {total_time:.2f} giây ({start_sys_time_all} → {end_sys_time_all}).")


def is_dark(image_path, brightness_threshold):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False

    avg_brightness = np.mean(image)

    return avg_brightness < brightness_threshold


def filter_dark_images(dataset_folder=os.path.join(config.ROOT_DIR, "PRE_DATASET"), brightness_threshold=20):
    removed_count = 0

    for movie_folder in os.listdir(dataset_folder):
        movie_path = os.path.join(dataset_folder, movie_folder)
        if not os.path.isdir(movie_path):
            continue

        for image_file in os.listdir(movie_path):
            image_path = os.path.join(movie_path, image_file)

            if is_dark(image_path, brightness_threshold):
                os.remove(image_path)
                removed_count += 1
                print(f"Đã xóa ảnh tối: {image_path}")


yolo_cfg = "./yolo/yolov3.cfg"
yolo_weights = "./yolo/yolov3.weights"
yolo_classes = "./yolo/coco.names"

net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
with open(yolo_classes, "r") as f:
    classes = [line.strip() for line in f.readlines()]

person_class_id = classes.index("person")  # Chỉ số của loại "person" trong COCO dataset


def detect_person(image_path):
    """Phát hiện người trong ảnh, trả về số lượng người."""
    image = cv2.imread(image_path)
    if image is None:
        return -1  # Không đọc được ảnh

    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    detections = net.forward(layer_names)

    person_count = 0
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == person_class_id and confidence > 0.5:  # Nếu phát hiện là người
                person_count += 1

    return person_count


def filter_images(dataset_folder=os.path.join(config.ROOT_DIR, "PRE_DATASET"), max_people=5):
    removed_count = 0

    for movie_folder in os.listdir(dataset_folder):
        movie_path = os.path.join(dataset_folder, movie_folder)
        if not os.path.isdir(movie_path):
            continue

        for image_file in os.listdir(movie_path):
            image_path = os.path.join(movie_path, image_file)
            person_count = detect_person(image_path)

            if person_count == -1:
                print(f"Không đọc được dữ liệu ảnh: {image_path}")
                continue

            if person_count == 0 or person_count > max_people:
                os.remove(image_path)
                removed_count += 1
                print(f"Đã xóa: {image_path} (Phát hiện {person_count} người)")

    print(f"Hoàn thành, đã xóa {removed_count} ảnh.")


# CHẠY CHƯƠNG TRÌNH
if __name__ == "__main__":
    process_all_videos()  # trích xuất frame từ video
    filter_dark_images()  # xóa ảnh tối
    filter_images()  # xóa ảnh ko có người hoặc quá nhiều người
