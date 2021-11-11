import cv2 
import numpy as np 

def draw_text_bbox(image, list_bboxes, color=(255, 0, 0), thickness=2, isClosed=True):
    for bbox in list_bboxes:
        pts = np.array(bbox)
        image = cv2.polylines(image, [pts], isClosed, color, thickness)

    return image
