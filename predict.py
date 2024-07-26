import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2

import pathlib

import sys

import numpy as np
import json

from paddleocr import PaddleOCR
from ultralytics import YOLO

import json     

from src.contours import  *
from src.ocr import get_pixel_real_size
from src.render import *

def get_mask(model, img):
    H, W, _ = img.shape

    results = model(img)
    mask = results[0].masks.data[0].cpu().numpy() * 255
    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, (W, H))

    return mask

def main(config):
    ocr = PaddleOCR(lang="en", use_angle_cls=False, show_log=False)

    middle_part_path = pathlib.Path(config['middle_part_path']).resolve()
    plate_model_path = pathlib.Path(config['plate_model_path']).resolve()
    model1 = YOLO(middle_part_path)
    model2 = YOLO(plate_model_path)


    image_path = pathlib.Path(config['image_path']).resolve()
    output_path = pathlib.Path(config['output_path']).resolve()

    if output_path.is_file():
        print("Output destination not a folder.")
        return None

    if not output_path.exists():
        output_path.mkdir()
    render = config['render']
    imgs = dict()
    if image_path.is_dir():
        for img_id in image_path.glob('**/*'):
            if img_id.is_file():
                img =  cv2.imread(str(img_id))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs[img_id.stem] =  img
    else:
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs[image_path.stem] = img
        image_path = image_path.parent

    
    res = []

    for key, img in imgs.items():
        im = cv2.imread(os.path.join(str(image_path), key+'.jpg'))

        plot = im.copy()

        le, u = get_pixel_real_size(ocr, im)
        mask1 = get_mask(model1, im)
        cv2.imwrite('mask.jpg', mask1)
        mask2 = get_mask(model2, im)
        mask2 = cv2.subtract(mask2,mask1)
        kernel = np.ones((5, 5))
        mask2 = cv2.erode(mask2, kernel, iterations=4)
        mask2 = cv2.dilate(mask2, kernel, iterations=4)

        main_object_con = n_max_contours(mask1)
        plate_part_cntrs = n_max_contours(mask2, n=2)

        quad = [approximte_contour(q) for q in plate_part_cntrs]
        rect = [bounding_rectangle(q) for q in quad]

        main_sides_rect = plate_width_line(*rect)
        t = np.linalg.norm(main_sides_rect[0][0] - main_sides_rect[0][1])
        
        (p11, p12), (p21, p22) = plate_width_line(*quad)
        c1, c2 = line_intersection_contur([(p11, p21), (p12, p22)], main_object_con)
        # c1, c2 = approximte_contour(c1, epsilon=0.001), approximte_contour(c2, epsilon=0.001)
        plot = cv2.polylines(plot, [c1], False, (255, 0, 255), 5)
        plot = cv2.polylines(plot, [c2], False, (255, 0, 255), 5)
        plot = cv2.line(plot, p11, p21, (255, 255, 0), 5)
        plot = cv2.line(plot, p12, p22, (255, 255, 0), 5)
        res_d = []
        for c, l in zip((c1, c2), ((p11, p21), (p12, p22))):
            dist, _, p1, p2 = find_deviation_peaks(l, c, 0.0005 * t)
            res_d.append(dist[0])
            if p1 is not None:
                for pr in zip(p1, p2):
                    plot = render_line(plot, pr, le, u)


        plot = render_line(plot, main_sides_rect[0], le, u)

        misallignment = calculate_bias(*main_sides_rect) * le
        
        plot = cv2.putText(plot, f'misalignment: %.2f' % misallignment + u, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
         
        res.append({
            "key": key,
            "h1": res_d[0] * le,
            "h2": res_d[1] * le,
            "t": t * le,
            "misallignment": misallignment
        })
        cv2.imwrite(os.path.join(str(output_path), key + '.jpg'), plot)
    with open("res.json", "w") as f:
        json.dump(res, f, indent=2)

if __name__ == '__main__':
    config_path = pathlib.Path('config.json').resolve()

    if not config_path.exists():
        print("Specified config file does not exists")

    with open(str(config_path), 'r') as json_data:
        config = json.loads(json_data.read())
        main(config)