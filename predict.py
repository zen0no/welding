import os

import torch

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
from src.gost import check_gosts
import pandas as pd

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
    output_masked = output_path / "masked"
    output_rendered = output_path / "rendered"
    output_result = output_path / "props.csv"
    output_gosts = output_path / "gosts.csv"


    output_masked.mkdir(parents=True, exist_ok=True)
    output_rendered.mkdir(parents=True, exist_ok=True)

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
    gosts = []

    for key, img in imgs.items():
        try:
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
            plot = draw_line(plot, p11, p21, (255, 255, 0), 5)
            plot = draw_line(plot, p12, p22, (255, 255, 0), 5)
            res_d = []
            for c, l in zip((c1, c2), ((p11, p21), (p12, p22))):
                dist, _, p1, p2 = find_deviation_peaks(l, c, 0.0005 * t)
                res_d.append(dist)
                if p1 is not None:
                    for pr in zip(p1, p2):
                        plot = render_line(plot, pr, le, u)

            plot = render_line(plot, main_sides_rect[0], le, u)

            misalignment = calculate_bias(*main_sides_rect) * le

            plot = cv2.putText(plot, f'misalignment: %.2f' % misalignment + u, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)

            b_downer = (np.linalg.norm(p11 - p21) ** 2 - calculate_projection_line_width(main_sides_rect[0], (p11, p21)) ** 2) ** 0.5
            b_upper = (np.linalg.norm(p12 - p22) ** 2 - calculate_projection_line_width(main_sides_rect[0], (p12, p22)) ** 2) ** 0.5

            res.append({
                "key": key,
                "he": res_d[1][0] * le,
                "hs": res_d[1][1] * le,
                "hg": res_d[0][0] * le,
                "hp": res_d[0][1] * le,
                "t": t * le,
                "hm": misalignment,
                "b_upper": b_upper * le,
                "b_downer": b_downer * le,
            })
            gosts.append(check_gosts(res[-1]))
            cv2.imwrite(str(output_rendered / f'{key}.jpg'), plot)
            cv2.imwrite(str(output_masked / f'{key}.jpg'), render_mask(img, mask1))
        except cv2.error as e:
            print(f"error while proceeding {key}")
            print(e)
    res_df = pd.DataFrame(res)
    res_df.to_csv(output_result)

    gosts_df = pd.DataFrame(gosts)
    gosts_df.to_csv(output_gosts)


if __name__ == '__main__':
    config_path = pathlib.Path('config.json').resolve()

    if not config_path.exists():
        print("Specified config file does not exists")

    with open(str(config_path), 'r') as json_data:
        config = json.loads(json_data.read())
        main(config)