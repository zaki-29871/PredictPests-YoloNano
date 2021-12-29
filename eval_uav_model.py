import torch
import pytorch_model.yolo_nano
import torch.optim as optim
import utils.stats
import utils.dataset
import utils.profile
from torch.utils.data import DataLoader
import tools
import numpy as np

image_size = 1536
device = 'cuda'
version = None
seed = 0
batch = 1

profile = utils.profile.YoloNanoPaper()

model = profile.load_model(num_classes=1, image_size=image_size, version=version)[1].to(device)
version, loss_history = profile.load_history(version)
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

test_dataset = utils.dataset.DirtyBugsUAV(image_size=image_size)

test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

print('Number of testing data:', len(test_dataset))
print(f'CUDA abailable cores: {torch.cuda.device_count()}')
print(f'Batch: {batch}')
print('Using model:', profile)
print('Using dataset:', test_dataset)
print('Network image size:', image_size)
print('Number of parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))

model.eval()
for batch_index, (X,) in enumerate(test_loader):
    tools.tic()
    X = X.to(device)

    with torch.no_grad():
        detections = model.forward(X)
        detections = utils.stats.non_max_suppression(detections.cpu(), 0.9, 0, use_label_match=False)

        time = utils.timespan_str(tools.toc(True))

        X = X[0].permute(1, 2, 0).data.cpu().numpy() * 255
        X = X.astype('uint8')

        utils.plot_detections(X, detections[0], root=r'D:\Dataset\YoloNano\uav_detection',
                              class_transformer=utils.dataset.DirtyBugsUAV.get_class_transformer(),
                              save_file_name=f'{batch_index}.png')

        print(f'[{batch_index + 1}/{len(test_loader)} {time}]')
