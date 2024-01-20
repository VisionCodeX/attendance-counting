from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
from get_points import Points
from shapely.geometry import Polygon
from shapely.geometry.point import Point
import numpy as np
from datetime import datetime

def main(weights, source, device, save):
    check_user_in_area = {"is_teacher":False, "teacher_id": None}

    model = YOLO(weights)
    model = model.to("cuda") if device == '0' else model.to("cpu")
    cap = cv2.VideoCapture(source)
    
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Define region points
    region_points = Points("points/lines.csv", "points/entrance_poly.json").get_polygon()
    teacher_are_point = Polygon(Points("points/lines.csv", "points/polygon.json").get_polygon())
    teacher_are_cords = np.array(teacher_are_point.exterior.coords, dtype=np.int32)
    already_inside = {}

    # Video writer
    if save:
        video_writer = cv2.VideoWriter("output_of_ultly.mp4",
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            fps,
                            (w, h))

    # Init Object Counter
    counter = object_counter.ObjectCounter()
    counter.set_args(view_img=False,
                    reg_pts=region_points,
                    classes_names=model.names,
                    draw_tracks=True,
                    region_thickness=2,
                    count_reg_color=(46, 204, 113),
                    line_dist_thresh=10)

    while cap.isOpened():
        success, im0 = cap.read()

        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        tracks = model.track(im0, persist=True, show=False, classes=[0])

        cv2.polylines(im0, [teacher_are_cords], isClosed=True, color=(255, 42, 4), thickness=2)
        im0 = counter.start_counting(im0, tracks)
        
        in_count = counter.in_counts
        out_count = counter.out_counts

        boxes = tracks[0].boxes.xyxy
        track_ids = tracks[0].boxes.id

        current_count = [i for i in tracks[0].boxes.cls if i.item() == 0]
        if in_count == 0 and out_count == 0:
            already_inside[len(current_count)] = 0

        is_teacher = {'is_teacher': [], 'teacher_id': []}

        for box, track_id in zip(boxes, track_ids):
            x_center = int((box[0] + box[2]) / 2)
            y_center = int((box[1] + box[3]) / 2)
            is_teacher_area = teacher_are_point.contains(Point((x_center, y_center)))
            is_teacher['is_teacher'].append(is_teacher_area)
            is_teacher['teacher_id'].append(track_id)

            
        cv2.putText(im0, f"Teacher     : {1 if any(is_teacher['is_teacher']) else 0}", (1600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(im0, f"CurrentCount: {len(current_count)}", (1600, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(im0, f"AlreadyInside: {len(already_inside)}", (1600, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("object_counting_output", im0)
        if save:
            video_writer.write(im0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if save:
        video_writer.release() 
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(weights="models/yolov8x.pt", source="data/videos/video.mp4", device="0", save=False)