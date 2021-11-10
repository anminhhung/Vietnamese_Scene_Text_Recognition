import numpy as np
import cv2

def crop_text_area(image, list_points):
    polygon = np.array(list_points)
    rect = cv2.boundingRect(polygon)
    x, y, w, h = rect
    croped = image[y:y+h, x:x+w].copy()

    polygon = polygon - polygon.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [polygon], -1, (255, 255, 255), -1, cv2.LINE_AA)

    dst = cv2.bitwise_and(croped, croped, mask=mask)

    bg = np.ones_like(croped, np.uint8)*255
    cv2.bitwise_not(bg, bg, mask=mask)
    dst2 = bg+ dst

    return dst2

def create_output_file(submit_file_path, bbox, text):
    with open(submit_file_path, "a+") as f:
        content = "{},{},{},{},{},{},{},{},{}".format(
                bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1], bbox[2][0], bbox[2][1], bbox[3][0], bbox[3][1], text)
        f.write("{}\n".format(content))