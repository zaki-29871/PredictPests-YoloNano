import torch
import pytorch_model.yolo_nano
import torch.optim as optim
import utils.stats
import utils.dataset
import utils.profile
from torch.utils.data import DataLoader
import tools
import numpy as np
import matplotlib.pyplot as plt

image_size = 1248
device = 'cuda'
version = 29  # 29
seed = 0
batch = 1
gaussian_blur_kernel = None
dataset = utils.dataset.DirtyBugs
profile = utils.profile.YoloNano()

if isinstance(profile, utils.profile.YoloNano):
    model = profile.load_model(num_classes=2, image_size=image_size, version=version)[1].to(device)
elif isinstance(profile, utils.profile.YoloNanoPaper):
    model = profile.load_model(num_classes=1, image_size=image_size, version=version)[1].to(device)
else:
    raise Exception('Unknown model')

class_transformer = dataset.get_class_transformer()

if dataset == utils.dataset.DirtyBugs:
    dataset = utils.dataset.DirtyBugs(image_size=image_size, ignore=True, gaussian_blur_kernel=gaussian_blur_kernel)
elif dataset == utils.dataset.DirtyBugsPaper:
    dataset = utils.dataset.clean_dirty_bugs_paper(utils.dataset.DirtyBugsPaper(image_size=image_size))
else:
    raise Exception('Unknown dataset')

test_dataset = utils.dataset.random_split(dataset, seed=seed, train_ratio=0.9)[1]

# test_dataset = utils.dataset.random_subset(test_dataset, 2, seed=0)

test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False,
                         collate_fn=utils.dataset.DirtyBugs.collate_fn)

print('Number of testing data:', len(test_dataset))
print(f'CUDA abailable cores: {torch.cuda.device_count()}')
print(f'Batch: {batch}')
print('Using model:', profile)
print('Using dataset:', dataset)
print('Network image size:', image_size)
print('Number of parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))

labels = []
sample_matrics = []
batch_precision_list = []

model.eval()
for batch_index, (X, targets) in enumerate(test_loader):
    tools.tic()
    X = X.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        detections = model.forward(X)
        detections = utils.stats.non_max_suppression(detections.cpu(), 0.9, 0.2, use_label_match=False)

        labels += targets[:, 1].tolist()
        targets[:, 2:] = utils.stats.xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= image_size

        batch_metrics = utils.stats.get_batch_statistics(detections, targets.cpu(), iou_threshold=0.5)
        sample_matrics += batch_metrics

        if len(batch_metrics) > 0:
            batch_precision = batch_metrics[0][0].sum() / batch_metrics[0][0].shape[0]
            batch_recall = batch_metrics[0][1].sum() / batch_metrics[0][1].shape[0]
            if batch_recall + batch_precision > 0:
                f1_score = 2 * (batch_recall * batch_precision) / (batch_recall + batch_precision)
            else:
                f1_score = 0

            X = X[0].permute(1, 2, 0).data.cpu().numpy() * 255
            X = X.astype('uint8')

            # utils.plot_detections(X, detections[0], root=r'D:\Dataset\YoloNano\detection',
            #                       class_transformer=class_transformer,
            #                       save_file_name=f'{batch_index}_{f1_score:.1%}.png')
        else:
            batch_precision = 0
            batch_recall = 0
            f1_score = 0

        time = utils.timespan_str(tools.toc(True))

        print(
            f'[{batch_index + 1}/{len(test_loader)} {time}] {batch_precision:.1%} {batch_recall:.1%} {f1_score:.1%}')
        batch_precision_list += [batch_precision]

true_positives_prediction, true_positives_annotation, pred_scores, pred_labels = [np.concatenate(x, 0) for x in
                                                                                  list(zip(*sample_matrics))]

precision = sum(true_positives_prediction) / len(true_positives_prediction)
recall = sum(true_positives_annotation) / len(true_positives_annotation)
f1_score = 2 * (recall * precision) / (recall + precision)

print(f'precision: {precision:.1%}')
print(f'recall: {recall:.1%}')
print(f'f1_score: {f1_score:.1%}')

plt.scatter(np.arange(0, len(batch_precision_list)), batch_precision_list)
plt.axhline(0, color='k', linestyle='--')
plt.axhline(1, color='k', linestyle='--')
plt.ylim([-0.1, 1.1])
# plt.show()
