import csv
import json
import numpy as np
class Points:

    def __init__(self, line_path, polygon_path):
        self.line_path = line_path
        self.polygon_path = polygon_path

    def get_lines(self):
        """
        Get points from a csv file

        example csv file:
        label, x1, y1, x2, y2, image_path, width, height
        """
        with open(self.line_path, 'r') as f:
            data = csv.reader(f)
            xyxy = list(map(int, list(data)[0][1:5]))
            return np.array(xyxy)

    def get_polygon(self):
        """
        Get points from a JSON file

        example JSON file:

        {
            "info": {
                "description": "Image containing a polygon",
            },
            "images": [
                {
                    "id": 0,
                    "file_name": "",
                    "width": 0,
                    "height": 0
                }
            "annotations": [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 0,
                    "iscrowd": 0,
                    "segmentation": []
                }
            ]
        }
                        
        """

        with open(self.polygon_path, 'r') as f:
            data = json.load(f)

            points = data['annotations'][0]['segmentation'][0]
            points = list(map(int, points))
            return np.array(points).reshape(-1, 2)
        

if __name__ == "__main__":
    points = Points('points/lines.csv', 'points/polygon.json')
    print(points.get_lines())
    print(points.get_polygon())