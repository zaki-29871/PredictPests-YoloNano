import utils.dataset
from torch.utils.data import DataLoader, Subset
import utils
import numpy as np

image_size = 1024
gaussian_blur_kernel = None
dataset = utils.dataset.DirtyBugsPaper

class_transformer = dataset.get_class_transformer()

if dataset == utils.dataset.DirtyBugs:
    train_dataset = utils.dataset.DirtyBugs(image_size=image_size, ignore=True,
                                            gaussian_blur_kernel=gaussian_blur_kernel)
elif dataset == utils.dataset.DirtyBugsPaper:
    train_dataset = utils.dataset.clean_dirty_bugs_paper(utils.dataset.DirtyBugsPaper(image_size=image_size))
else:
    raise Exception('Unknown dataset')

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False,
                          collate_fn=utils.dataset.DirtyBugs.collate_fn)

for batch_index, (X, Y) in enumerate(train_loader):
    print('batch', batch_index)
    utils.plot_ground_true(X, Y, class_transformer,
                           save_file_name=f'{batch_index}.png',
                           root=r'D:\Dataset\YoloNano\display')
