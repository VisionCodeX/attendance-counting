import supervision as sv
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import argparse
import numpy as np
# import Point class
from supervision.draw.color import Color, ColorPalette
from supervision.geometry.core import Point, Rect, Vector

from get_points import Points

def annotator(detections, labels, frame, line_counter):
    """
    Annotate the frame with detections and labels
    """

    # create box annotator, it will draw a box for each detection
    box_annotator = sv.BoxAnnotator(
        color=ColorPalette.default(), 
        thickness=2, 
        text_thickness=2, 
        text_scale=1.5
        )
    # create line annotator, it will draw a line for each detection
    line_annotator = sv.LineZoneAnnotator(
        thickness=3, 
        text_thickness=3, 
        text_scale=2)
    # create trace annotator, it will draw a trace for each detection
    trace_annotator = sv.TraceAnnotator(
        color=sv.Color.red()
        )

    # annotate the frame
    frame = line_annotator.annotate(
        frame=frame,
        line_counter=line_counter
        )

    # annotate the frame, it will draw a box for each detection
    frame = box_annotator.annotate(
        scene=frame, 
        detections=detections, 
        labels=labels)
    # annotate the frame, it will draw a trace for each detection
    frame = trace_annotator.annotate(
        scene=frame, 
        detections=detections
        )

    return frame

def main(weights, source, device, save):
    """
    Main function to run the program
    """
    # load the model
    model = YOLO(f'{weights}')
    # move the model to the device
    model.to('cuda') if device == '0' else model.to('cpu')
    # Extract classes names
    names = model.model.names
    
    # create a tracker object
    tracker = sv.ByteTrack()

    # create LineZone object, it will count the number of detections that cross the line
    x1, y1, x2, y2 = Points("points/lines.csv", "points/polygon.json").get_lines()
    line_counter = sv.LineZone(start=Point(x1, y1), end=Point(x2, y2))

    zone = sv.PolygonZone(polygon=Points("points/lines.csv", "points/polygon.json").get_polygon(), frame_resolution_wh=(1920, 1080))
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.white(), thickness=3, text_thickness=3, text_scale=2)

    # create a video capture object
    videocapture = cv2.VideoCapture(source)
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*'XVID')

    
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    video_writer = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

    while videocapture.isOpened():

        ret, frame = videocapture.read()
        if not ret:
            break
        
        # predict the frame
        result = model(frame, classes=[0])[0]
        # convert the result to detections
        detections = sv.Detections.from_ultralytics(result)
        # update the tracker with the detections
        detections = tracker.update_with_detections(detections)

        # create a list of labels for each detection
        labels = [f"#{tracker_id} {model.names[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, tracker_id in detections]

        # annotate the frame
        # trigger the line counter, it will return a tuple of two boolean arrays
        # the first array will be True for each detection that crossed the line from the outside to the inside
        line_counter.trigger(
            detections=detections
            )
        zone.trigger(
            detections=detections
            )
        # print(zone.current_count,zone.polygon)
        frame = zone_annotator.annotate(scene=frame)

        frame = annotator(detections, labels, frame, line_counter)
        
        if save:
            video_writer.write(frame)
        # show the frame
        cv2.imshow('annotated_frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    if save:
        video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--weights", type=str, default="yolov8s.pt", help="path to weights file")
    argparser.add_argument("--source", type=str, default="data/videos/video.mp4", help="path to source")
    argparser.add_argument("--save", type=bool, action=argparse.BooleanOptionalAction, default=False, help="save the video")
    args = argparser.parse_args()

    weights=args.weights
    source=args.source
    save=args.save

    main(weights, source, device='0', save=save)
