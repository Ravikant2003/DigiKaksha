import os
import cv2
import sys
import face_recognition
from zipfile import ZipFile
from urllib.request import urlretrieve
import csv
import datetime

def download_and_unzip(url, save_path):
    print("Downloading and extracting assets....", end="")
    urlretrieve(url, save_path)
    try:
        with ZipFile(save_path) as z:
            z.extractall(os.path.split(save_path)[0])
        print("Done")
    except Exception as e:
        print("\nInvalid file.", e)

def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
            img_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0])

    return known_face_encodings, known_face_names

def write_attendance(attendance):
    attendance_file = "attendance.csv"
    with open(attendance_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Attendance", "Timestamp"])
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for name, present in attendance.items():
            writer.writerow([name, "Present" if present else "Absent", timestamp])

def main():
    URL = r"https://www.dropbox.com/s/efitgt363ada95a/opencv_bootcamp_assets_12.zip?dl=1"
    asset_zip_path = os.path.join(os.getcwd(), "opencv_bootcamp_assets_12.zip")

    if not os.path.exists(asset_zip_path):
        download_and_unzip(URL, asset_zip_path)

    known_faces_dir = "known_faces"
    known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

    # Try to use an external camera first (index 1)
    source = cv2.VideoCapture(1)
    if not source.isOpened():
        print("External camera not found. Switching to built-in camera.")
        source = cv2.VideoCapture(0)

    win_name = "Camera Preview"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
    in_width = 300
    in_height = 300
    mean = [104, 117, 123]
    conf_threshold = 0.7

    attendance = {name: False for name in known_face_names}

    while cv2.waitKey(1) != 27:
        has_frame, frame = source.read()
        if not has_frame:
            break
        frame = cv2.flip(frame, 1)
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)
        net.setInput(blob)
        detections = net.forward()

        face_locations = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
                y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
                x_right_top = int(detections[0, 0, i, 5] * frame_width)
                y_right_top = int(detections[0, 0, i, 6] * frame_height)

                face_locations.append((y_left_bottom, x_right_top, y_right_top, x_left_bottom))

                cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0))
                label = "Confidence: %.4f" % confidence
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(
                    frame,
                    (x_left_bottom, y_left_bottom - label_size[1]),
                    (x_left_bottom + label_size[0], y_left_bottom + base_line),
                    (255, 255, 255),
                    cv2.FILLED,
                )
                cv2.putText(frame, label, (x_left_bottom, y_left_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = min(range(len(face_distances)), key=lambda i: face_distances[i])
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                attendance[name] = True

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow(win_name, frame)

    source.release()
    cv2.destroyAllWindows()

    write_attendance(attendance)

if __name__ == "__main__":
    main()
