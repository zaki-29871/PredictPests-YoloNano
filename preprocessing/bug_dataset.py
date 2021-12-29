# import yolo_nano
from PIL import Image, ImageDraw
import numpy as np
import os
import json
import pickle
import tools
import utils.dataset

dataset = utils.dataset.DirtyBugsPaper
root = dataset.ROOT
os.makedirs(os.path.join(root, 'pkl/target'), exist_ok=True)

for i in range(0, 752):
    targets = utils.dataset.DirtyBugs.parse_annotation(root, i, dataset.get_class_transformer())
    if targets is not None:
        targets = np.array(targets, dtype=np.float32)
        tools.save(targets, os.path.join(root, 'pkl/target', f'{i}.pkl'))