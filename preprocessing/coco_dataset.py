# import yolo_nano
from PIL import Image, ImageDraw
import numpy as np
import os
import json
import pickle
import tools

ROOT = 'D:\Dataset\YoloNano\datasets\coco'
file = os.path.join(ROOT, 'annotations/instances_train2017.json')

def load_annotation(filename):
    # dict
    # json
    cache = os.path.join(ROOT, 'cache/instances_train2017.pkl')
    os.makedirs(os.path.join(ROOT, 'cache'), exist_ok=True)
    if os.path.exists(cache):
        print('using cache')
        with open(cache, 'rb') as cache_file:
            annotation = pickle.load(cache_file)
    else:
        with open(filename, 'r') as file:
            annotation = json.loads(file.readline())

        with open(cache, 'wb') as cache_file:
            pickle.dump(annotation, cache_file)

    return annotation

annotation = load_annotation(file)

print(annotation.keys())

X = {}
Y = []
# Y = {}
C = {}

for x in annotation['images']:
    X[x['id']] = x

for y in annotation['annotations']:
    if int(y['image_id']) in [49, 92]:
        Y.append([y['image_id'], y['category_id'], *y['bbox']])

for c in annotation['categories']:
    C[c['id']] = (c['supercategory'], c['name'])

Y = np.array(Y)
print('categories:', len(C.keys()))
print('Y shape:', Y.shape)

os.makedirs(os.path.join(ROOT, 'pkl'), exist_ok=True)
tools.save(C, os.path.join(ROOT, 'pkl/categories.pkl'))
tools.save(Y, os.path.join(ROOT, 'pkl/Y.pkl'))

# print(C[Y[92]['category_id']])
# print(X[92])
# print(Y[92])
# print(annotation.keys())

# image = Image.open(os.path.join(ROOT, f'images/train2017/{92:012d}.jpg'))
# img1 = ImageDraw.Draw(image)
# img1.rectangle([(125.7, 0.96), (125.7+376.14, 0.96+292.66)], outline="red")
# image.show()