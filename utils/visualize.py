import cv2 
import numpy as np 

def write_text(image, result_text, point, color=(0, 0, 255), thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    image = cv2.putText(image, result_text, point, font, fontScale, color, thickness, cv2.LINE_AA)

    return image

def draw_text_bbox(image, list_bboxes, list_result_text, color=(255, 0, 0), thickness=2, isClosed=True):
    for bbox, result_text in zip(list_bboxes, list_result_text):
        pts = np.array(bbox)
        image = cv2.polylines(image, [pts], isClosed, color, thickness)
        image = write_text(image, result_text, (bbox[0][0], bbox[0][1]))

    return image