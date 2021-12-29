import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import utils.dataset
import os
import torch


class TargetTransformer:
    def __init__(self):
        self.origin_image_size = None
        self.small_image_size = None
        self.padding = None
        self.target_image_size = None

    def x1y1_to_xyhw(self, x):
        """
        :param x: [idx, class_id, x1, y1, h, w]
            x1, y1 is the coordinate of the left top corner
            x1, y1, h, w in pixel unit
        :return: [idx, class_id, cx, cy, h, w] in yolo format
        """
        h, w = self.origin_image_size
        y = np.zeros(x.shape, dtype=np.float32)
        y[:, :2] = x[:, :2]
        y[:, 2] = (x[:, 2] + x[:, 4] / 2) / w  # center x, normalize by width
        y[:, 3] = (x[:, 3] + x[:, 5] / 2) / h  # center y, normalize by height
        y[:, 4] = x[:, 4] / w
        y[:, 5] = x[:, 5] / h
        return y

    def padding_transform(self, x):
        """
        :param x: [idx, class_id, cx, cy, h, w] in yolo format
            cx, cy is the coordinate of the center
            cx, w are normalized by width
            cy, h are normalized by height
        :return: [idx, class_id, cx, cy, h, w] in letter box image with yolo format
        """
        sh, sw = self.small_image_size
        py, px = self.padding
        th, tw = self.target_image_size

        y = np.zeros(x.shape, dtype=np.float32)
        y[:, :2] = x[:, :2]

        # Consider padding in column axis
        y[:, 2] = (x[:, 2] * sw + px) / tw
        y[:, 4] = (x[:, 4] * sw) / tw

        # Consider padding in row axis
        y[:, 3] = (x[:, 3] * sh + py) / th
        y[:, 5] = (x[:, 5] * sh) / th

        return y

    def detection_inverse_transform(self, detection: torch.Tensor):
        """
        :param detection: [x1, y1, x2, y2, object_conf, class_score, class_pred]
            x1, y1, x2, y2 in pixel unit, in letter box image
        :return: [x1, y1, x2, y2, object_conf, class_score, class_pred]
            x1, y1, x2, y2 in pixel unit, in original image
        """
        h, w = self.origin_image_size
        sh, sw = self.small_image_size
        py, px = self.padding

        y = detection.new_zeros(detection.shape)
        y[:, 4:7] = detection[:, 4:7]  # object_conf, class_score, class_pred

        # padding inverse transform
        y[:, 0] = (detection[:, 0] - px) / sw  # x1
        y[:, 1] = (detection[:, 1] - py) / sh  # y1
        y[:, 2] = (detection[:, 2] - px) / sw  # x2
        y[:, 3] = (detection[:, 3] - py) / sh  # y2

        # change to original image pixel coordinates
        y[:, 0] = y[:, 0] * w  # x1
        y[:, 1] = y[:, 1] * h  # y1
        y[:, 2] = y[:, 2] * w  # x2
        y[:, 3] = y[:, 3] * h  # y2

        return y


def letterbox_image(image, targets=None, target_size=416):
    # targets: [idx, class_id, x1, y1, h, w]
    # x1, y1 is the coordinate of the left top corner
    # x1, y1, h, w in pixel unit
    # idx is used to associate the bounding boxes with its image
    # skip images without bounding boxes (mainly because coco has unlabelled images)
    if targets is not None and len(targets) == 0:
        targets = None

    h, w = image.shape[:2]
    target_transformer = TargetTransformer()
    target_transformer.origin_image_size = (h, w)
    target_transformer.target_image_size = (target_size, target_size)

    if targets is not None:
        targets = target_transformer.x1y1_to_xyhw(targets)

    if h > w:
        image_new = np.zeros((*target_transformer.target_image_size, 3), dtype=np.uint8)
        ratio = target_size / h
        small_img = cv2.resize(image, None, fx=ratio, fy=ratio)
        padding_width = int((target_size - small_img.shape[1]) / 2)
        image_new[:, padding_width:(padding_width + small_img.shape[1]), :] = small_img[:, :, :]

        target_transformer.small_image_size = small_img.shape[0:2]
        target_transformer.padding = (0, padding_width)

    elif w >= h:
        image_new = np.zeros((*target_transformer.target_image_size, 3), dtype=np.uint8)
        ratio = target_size / w
        small_img = cv2.resize(image, None, fx=ratio, fy=ratio)
        padding_height = int((target_size - small_img.shape[0]) / 2)
        image_new[padding_height:(padding_height + small_img.shape[0]), :, :] = small_img[:, :, :]

        target_transformer.small_image_size = small_img.shape[0:2]
        target_transformer.padding = (padding_height, 0)

    if targets is None:
        return image_new, target_transformer
    else:
        targets = target_transformer.padding_transform(targets)
        return image_new, targets, target_transformer


def plot_ground_true(X, Y, class_transformer: utils.dataset.ClassTransformer, root, index=0, save_file_name=None):
    # Y: [idx, class_id, cx, cy, h, w]
    # idx is used to associate the bounding boxes with its image
    # cx, w are normalize by image width
    # cy, h are normalize by image height

    X = X[index]
    X = X.permute(1, 2, 0).data.cpu().numpy() * 255
    X = X.astype('uint8')
    Y = Y[Y[:, 0] == index].data.cpu().numpy()

    X_img = Image.fromarray(X)
    drawer = ImageDraw.Draw(X_img)

    for t in Y:
        class_id, cx, cy, w, h = t[1:]
        x = (cx - w / 2) * X.shape[1]
        y = (cy - h / 2) * X.shape[0]
        w = w * X.shape[1]
        h = h * X.shape[0]
        # print('target: class_id, x, y, w, h =', int(class_id), x, y, w, h)
        class_id = int(class_id)
        class_name = class_transformer.id_to_name[class_id]
        if class_id == 0:
            drawer.text((x, y - 20), class_name, font=ImageFont.truetype("arial", 14), fill=(255, 0, 0, 255))
            drawer.rectangle([x, y, x + w, y + h], outline="red", width=2)

        elif class_id == 1:
            drawer.text((x, y - 20), class_name, font=ImageFont.truetype("arial", 14), fill=(0, 0, 255, 255))
            drawer.rectangle([x, y, x + w, y + h], outline="blue", width=2)

    if save_file_name is None:
        X_img.show()
    else:
        os.makedirs(root, exist_ok=True)
        X_img.save(os.path.join(root, save_file_name))


def plot_detections(image: np.ndarray, detections, class_transformer: utils.dataset.ClassTransformer,
                    root, print_text=False, save_file_name=None):
    assert image.dtype == 'uint8'

    image = Image.fromarray(image)
    drawer = ImageDraw.Draw(image)

    if detections is not None:
        for detect in detections:
            x1, y1, x2, y2, object_conf, class_score, class_pred = detect[0:7].cpu().data.numpy()
            class_id = int(class_pred)
            class_name = class_transformer.id_to_name[class_id]
            if print_text:
                print('detect class:', class_name)
                print(f'\tx1 = {x1}')
                print(f'\ty1 = {y1}')
                print(f'\tx2 = {x2}')
                print(f'\ty2 = {y2}')
                print(f'\tobject_conf = {object_conf}')
                print(f'\tclass_score = {class_score}')

            class_name = class_name + f' ({object_conf:.0%}, {class_score:.0%})'
            if class_id == 0:
                drawer.text((x1, y1 - 20), class_name, font=ImageFont.truetype("arial", 14), fill=(255, 0, 0, 255))
                drawer.rectangle([x1, y1, x2, y2], outline="red", width=2)

            elif class_id == 1:
                drawer.text((x1, y1 - 20), class_name, font=ImageFont.truetype("arial", 14), fill=(0, 0, 255, 255))
                drawer.rectangle([x1, y1, x2, y2], outline="blue", width=2)

    if save_file_name is None:
        image.show()
    else:
        os.makedirs(root, exist_ok=True)
        image.save(os.path.join(root, save_file_name))


def version_code(file):
    a = file.index('-')
    b = file.index('.')
    return int(file[a + 1:b])


def get_latest_version(file_path):
    version_codes = [version_code(x) for x in os.listdir(file_path)]
    if len(version_codes) > 1:
        version_codes.sort()
        return version_codes[-1]
    else:
        return None


def timespan_str(timespan):
    total = timespan.seconds
    second = total % 60 + timespan.microseconds / 1e+06
    total //= 60
    minute = int(total % 60)
    total //= 60
    return f'{minute:02d}:{second:05.2f}'
