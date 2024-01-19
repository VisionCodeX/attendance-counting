from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
from get_points import Points

model = YOLO("models/yolov8x.pt")
cap = cv2.VideoCapture("data/videos/video.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points
# region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]
region_points = Points("points/lines.csv", "points/entrance_poly.json").get_polygon()
# Video writer
video_writer = cv2.VideoWriter("object_counting_output.mp4",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h))

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=region_points,
                 classes_names=model.names,
                 draw_tracks=True)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False)

    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()