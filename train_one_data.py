import torch.optim as optim
import torch
import pytorch_model.yolo_nano
import tools
import os
from PIL import Image, ImageDraw
import numpy as np
import torch.optim as optim
import utils.stats
import utils
import cv2

ROOT = 'D:\Dataset\YoloNano\datasets\coco'
image_size = 416
device = 'cuda'
train, load_nn = 0, 1
nn_path = './nn/yolo.nn'

Y_pkl = tools.load(os.path.join(ROOT, 'pkl/Y.pkl'))[1:3]
X_img = Image.open(os.path.join(ROOT, f'images/train2017/{int(Y_pkl[0, 0]):012d}.jpg'))
C = tools.load(os.path.join(ROOT, 'pkl/categories.pkl'))

X_img_np = np.array(X_img)
print('image height, width:', X_img_np.shape[0], X_img_np.shape[1])

X, target = utils.letterbox_image(X_img_np, Y_pkl, image_size)
X_img = Image.fromarray(X)

X = torch.tensor(X)
X = X.permute(2, 0, 1).unsqueeze(0).float() / 255

Y = torch.tensor(target).float()
print(Y)
Y[:, 0] = 0
print(Y.size())

print(f'height = {X.size(2)}, width = {X.size(3)}')
for class_id in Y[:, 1]:
    class_id = int(class_id)
    print(f'category: {class_id}, type={C[class_id]}')

X = X.to(device)
Y = Y.to(device)

model = pytorch_model.yolo_nano.YOLONano(num_classes=80, image_size=image_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

if load_nn:
    print(f'load model: {nn_path}')
    model.load_state_dict(torch.load(nn_path))

if train:
    for i in range(100):
        optimizer.zero_grad()
        # 10647 = (52*52 + 26*26 + 13*13) * 3
        loss, detections = model.forward(X, targets=Y)
        print(f'[{i + 1}] {loss: .3f}')
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), nn_path)

detections = model.forward(X)
# print(detections.size())

for detect in detections[0]:
    cx, cy, w, h, conf = detect[:5].cpu().data.numpy()
    if conf > 0.95:
        print('cx, cy, w, h, conf =', cx, cy, w, h, conf)

detections = utils.stats.non_max_suppression(detections.cpu(), 0.5, 0.4)

img1 = ImageDraw.Draw(X_img)

for t in target:
    class_id, cx, cy, w, h = t[1:]
    x = (cx - w / 2) * X.size(3)
    y = (cy - h / 2) * X.size(2)
    w = w * X.size(3)
    h = h * X.size(2)
    print('target: class_id, x, y, w, h =', int(class_id), x, y, w, h)
    img1.rectangle([x, y, x + w, y + h], outline="red", width=1)

for img_detect in detections:
    for detect in img_detect:
        # detect[0:7]
        x1, y1, x2, y2, object_conf, class_score, class_pred = detect[0:7].cpu().data.numpy()
        class_name = C[int(class_pred)]
        print('detect class:', class_name)
        print(f'\tx1 = {x1}')
        print(f'\ty1 = {y1}')
        print(f'\tx2 = {x2}')
        print(f'\ty2 = {y2}')
        print(f'\tobject_conf = {object_conf}')
        print(f'\tclass_score = {class_score}')
        img1.rectangle([x1, y1, x2, y2], outline="blue")

X_img.show()
