# Description: Run the model with ultralytics
python with_ultralytics.py --weights models/best.pt --source data/videos/video.mp4 --device 0 --save --output outputs/person_detection1.mp4