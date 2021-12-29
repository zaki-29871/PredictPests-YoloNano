import torch
import pytorch_model.yolo_nano
import torch.optim as optim
import utils.stats
import utils.dataset
import utils.profile
import utils
from torch.utils.data import DataLoader
import tools
import numpy as np
import os
import cv2

class YoloUAVDetector:
    def __init__(self, paper_image_size=1536, bug_image_size=1024, paper_model_version=200, bugs_model_version=40,
                 device='cuda', output_root=r'D:\Dataset\YoloNano\uav_detection'):
        self.paper_image_size = paper_image_size
        self.bug_image_size = bug_image_size
        self.paper_model_version = paper_model_version
        self.bugs_model_version = bugs_model_version
        self.device = device
        self.output_root = output_root

        self.paper_model = \
            utils.profile.YoloNanoPaper().load_model(num_classes=1, image_size=paper_image_size,
                                                     version=self.paper_model_version)[1].to(device)
        self.bugs_model = \
            utils.profile.YoloNano().load_model(num_classes=2, image_size=bug_image_size,
                                                version=self.bugs_model_version)[1].to(device)

        self.paper_model.eval()
        self.bugs_model.eval()

    def forward(self, uav_image):
        """
        :param uav_image: in bgr channel, numpy shape = (height, width, channel)
        :return: bug detections: [x1, y1, x2, y2, object_confidence, class_score, class_prediction] each row
        """
        X, target_transformer = utils.letterbox_image(uav_image, target_size=self.paper_image_size)
        X = torch.tensor(X)
        X = X.permute(2, 0, 1).float() / 255
        X = X.to(self.device).unsqueeze(0)

        # detect paper
        with torch.no_grad():
            paper_detections = self.paper_model.forward(X)
            paper_detections = utils.stats.non_max_suppression(paper_detections.cpu(), 0.9, 0, use_label_match=False)
            if paper_detections[0] is not None:
                paper_detections = target_transformer.detection_inverse_transform(paper_detections[0])

                # utils.plot_detections(uav_image, paper_detections, root=self.output_root,
                #                       class_transformer=utils.dataset.DirtyBugsUAV.get_class_transformer(),
                #                       save_file_name=f'{batch_index}_paper.png')
            else:
                paper_detections = None

        bug_detections = []
        if paper_detections is not None:
            for detect in paper_detections:
                x1, y1, x2, y2 = detect[0:4].cpu().data.numpy().astype('int')
                bug_image = uav_image[y1:y2, x1:x2, :]

                if bug_image.shape[0] == 0 or bug_image.shape[1] == 0:
                    continue

                X, target_transformer = utils.letterbox_image(bug_image, target_size=self.bug_image_size)
                X = torch.tensor(X)
                X = X.permute(2, 0, 1).float() / 255
                X = X.to(self.device).unsqueeze(0)

                # detect bugs
                with torch.no_grad():
                    bug_detect = self.bugs_model.forward(X)
                    bug_detect = utils.stats.non_max_suppression(bug_detect.cpu(), 0.9, 0.2, use_label_match=False)
                    if bug_detect[0] is not None:
                        bug_detect = target_transformer.detection_inverse_transform(bug_detect[0])
                        # utils.plot_detections(bug_image, bug_detect, root=self.output_root,
                        #                       class_transformer=utils.dataset.DirtyBugs.get_class_transformer(),
                        #                       save_file_name=f'{batch_index}_bug.png')

                        bug_detect[:, 0] = bug_detect[:, 0] + x1  # x1
                        bug_detect[:, 1] = bug_detect[:, 1] + y1  # y1
                        bug_detect[:, 2] = bug_detect[:, 2] + x1  # x2
                        bug_detect[:, 3] = bug_detect[:, 3] + y1  # y2

                        bug_detections += [bug_detect]

        if len(bug_detections) > 0:
            bug_detections = torch.cat(bug_detections, dim=0)
        else:
            bug_detections = None

        return bug_detections


# ROOT = r'D:\Dataset\YoloNano\UAVimg\20190515C1'
ROOT = r'D:\Dataset\YoloNano\UAVimg\\20190522'

detector = YoloUAVDetector()
file_list = os.listdir(ROOT)

print('Number of testing data:', len(file_list))
print('Using dataset:', ROOT)

for batch_index, filename in enumerate(file_list):
    tools.tic()
    X = cv2.imread(os.path.join(ROOT, filename))
    X = utils.dataset.rgb2bgr(X)

    # bug detections: [x1, y1, x2, y2, object_confidence, class_score, class_prediction] each row
    # class_prediction = 0 is fly
    # class_prediction = 1 is tetrigoidea
    # see utils.dataset.BugClassTransformer
    bug_detections = detector.forward(X)

    utils.plot_detections(X, bug_detections, root=detector.output_root,
                          class_transformer=utils.dataset.DirtyBugs.get_class_transformer(),
                          save_file_name=f'{batch_index}_full.png')

    time = utils.timespan_str(tools.toc(True))

    if bug_detections is None:
        print(f'[{batch_index + 1}/{len(file_list)} {time}] detect 0 bugs')
    else:
        print(f'[{batch_index + 1}/{len(file_list)} {time}] detect {bug_detections.size(0)} bugs')
