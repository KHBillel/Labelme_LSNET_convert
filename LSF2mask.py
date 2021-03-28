import json
import cv2
import numpy
import os
import numpy as np


CELL_HEIGHT = 32.0
CELL_WIDTH = 32.0

GRID_W = 80
GRID_H = 45

IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280

def colorize(proba, segs, threshold = 0.3, line_thickness = 3) :
    image = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), np.uint8)*255
    for i, row in enumerate(segs) :
        for j, cell in enumerate(row):
            if proba[i][j][0] >= threshold:
                x1 = int((j + cell[0]) * CELL_WIDTH)
                y1 = int((i + cell[1]) * CELL_HEIGHT)
                x2 = int((j + cell[2]) * CELL_WIDTH)
                y2 = int((i + cell[3]) * CELL_HEIGHT)
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=line_thickness)

    return image


if __name__ == '__main__':
    import base64
    with open(os.path.join(os.path.dirname(__file__),"full_dataset_test_lsnet/0.json")) as f :
        img_data = json.load(f)["imageData"]
        with open(os.path.join(os.path.dirname(__file__),"0.png"), "wb") as fh:
            fh.write(base64.b64decode(img_data))

    with open(os.path.join(os.path.dirname(__file__),"lsnet_format/proba_0.json")) as fproba :
        with open(os.path.join(os.path.dirname(__file__),"lsnet_format/seg_0.json")) as fseg :
            proba = json.load(fproba)
            segs = json.load(fseg)
            img = colorize(proba,segs)
            cv2.imwrite(os.path.join(os.path.dirname(__file__),"out.jpg"), img)

    img = cv2.imread(os.path.join(os.path.dirname(__file__),"0.png"))
    mask = cv2.imread(os.path.join(os.path.dirname(__file__),"out.jpg"))
    res = (0.7*img.astype(np.float32) + 0.3*mask.astype(np.float32)).astype(np.uint8)
    cv2.imshow("Result combi", res)
    cv2.waitKey(0)
    cv2.destroyWindow("Result combi")

