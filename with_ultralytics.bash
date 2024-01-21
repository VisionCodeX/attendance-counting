# Description: Run the model with ultralytics
python with_ultralytics.py --weights models/yolov8x_crowdhuman.pt --source data/videos/video.mp4 --device 0 --save --output outputs/person_crowdmodel1.mp4