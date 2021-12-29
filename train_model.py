import torch
import pytorch_model.yolo_nano
import torch.optim as optim
import utils.stats
import utils.dataset
import utils.profile
from torch.utils.data import DataLoader
import tools

image_size = 1024
device = 'cuda'
max_version = 50
version = None
seed = 0
batch = 4
gaussian_blur_kernel = 21

profile = utils.profile.YoloNano()

if isinstance(profile, utils.profile.YoloNano):
    model = profile.load_model(num_classes=2, image_size=image_size, version=version)[1].to(device)

elif isinstance(profile, utils.profile.YoloNanoPaper):
    model = profile.load_model(num_classes=1, image_size=image_size, version=version)[1].to(device)
else:
    raise Exception('Unknown model')

version, loss_history = profile.load_history(version)
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

# dataset = utils.dataset.clean_dirty_bugs_paper(utils.dataset.DirtyBugsPaper(image_size=image_size))
dataset = utils.dataset.DirtyBugs(image_size=image_size, ignore=True, gaussian_blur_kernel=gaussian_blur_kernel)

train_dataset, test_dataset = utils.dataset.random_split(dataset, seed=seed, train_ratio=0.9)

# train_dataset = utils.dataset.random_subset(train_dataset, 4)
# test_dataset = utils.dataset.random_subset(test_dataset, 4)

train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True,
                          collate_fn=utils.dataset.DirtyBugs.collate_fn)

test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=True,
                         collate_fn=utils.dataset.DirtyBugs.collate_fn)

print('Number of training data:', len(train_dataset))
print('Number of testing data:', len(test_dataset))
print(f'CUDA abailable cores: {torch.cuda.device_count()}')
print(f'Batch: {batch}')
print('Using model:', profile)
print('Using dataset:', dataset)
print('Network image size:', image_size)
print('Number of parameters: {:,}'.format(sum(p.numel() for p in model.parameters())))

for v in range(version, max_version + 1):
    train_loss = []
    test_loss = []

    print('Start training, version = {}'.format(v))
    model.train()
    for batch_index, (X, Y) in enumerate(train_loader):
        tools.tic()
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        # 10647 = (52*52 + 26*26 + 13*13) * 3
        loss, detections = model.forward(X, targets=Y)
        loss.backward()
        optimizer.step()
        # utils.plot_XY(X, Y)

        train_loss.append(float(loss))
        time = utils.timespan_str(tools.toc(True))
        print(f'[{v}/{max_version} - {batch_index + 1}/{len(train_loader)} {time}] loss = {loss: .3f}')

    train_loss = float(torch.tensor(train_loss).mean())

    print('Start testing, version = {}'.format(v))
    model.eval()
    for batch_index, (X, Y) in enumerate(test_loader):
        tools.tic()
        X = X.to(device)
        Y = Y.to(device)
        with torch.no_grad():
            loss, detections = model.forward(X, targets=Y)
            test_loss.append(float(loss))
            time = utils.timespan_str(tools.toc(True))
            print(f'[{v}/{max_version} - {batch_index + 1}/{len(test_loader)} {time}] val_loss = {loss: .3f}')

    test_loss = float(torch.tensor(test_loss).mean())
    print(f'Avg val_loss = {test_loss:.3f}')

    loss_history['train'].append(train_loss)
    loss_history['test'].append(test_loss)

    print('Start save model')
    profile.save_version(model, loss_history, v)

