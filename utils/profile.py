import pytorch_model.yolo_nano
import pytorch_model.yolo_nano_spp
import os
import utils
import tools
import torch
import torch.nn.functional as F
import torch.nn as nn

class Profile:
    def __init__(self):
        os.makedirs(self.version_file_path(), exist_ok=True)
        assert '-' not in str(self)

    def load_model(self, num_classes, image_size, version=None):
        self.model = self.get_model(num_classes, image_size)

        if version is None:
            print('Find latest version')
            version = utils.get_latest_version(self.version_file_path())

        if version is None:
            print('Can not find any version')
            version = 1
        else:
            print('Using version:', version)
            nn_file = self.model_file_name(version)

            if os.path.exists(nn_file):
                print('Load version model:', nn_file)
                self.model.load_state_dict(torch.load(nn_file))
            else:
                raise Exception(f'Cannot find neural network file: {nn_file}')

            version += 1

        return version, self.model

    def load_history(self, version=None):
        loss_history = {
            'train': [],
            'test': []
        }

        if version is None:
            print('Find latest version')
            version = utils.get_latest_version(self.version_file_path())

        if version is None:
            print('Can not find any version')
            version = 1
        else:
            print('Using version:', version)
            ht_file = self.history_file_name(version)

            if os.path.exists(ht_file):
                print('Load version history:', ht_file)
                loss_history = tools.load(ht_file)
            else:
                raise Exception(f'Cannot find history file: {ht_file}')

            version += 1

        return version, loss_history

    def save_version(self, model, history, version):
        torch.save(model.state_dict(), self.model_file_name(version))
        tools.save(history, self.history_file_name(version))

    def model_file_name(self, version):
        return os.path.join(self.version_file_path(), f'{self}-{version}.nn')

    def history_file_name(self, version):
        return os.path.join(self.version_file_path(), f'{self}-{version}.ht')

    def version_file_path(self):
        return f'./model/{self}'

    def get_model(self, num_classes, image_size):
        raise NotImplementedError()

    def __str__(self):
        return type(self).__name__

class YoloNano(Profile):
    def get_model(self, num_classes, image_size):
        return pytorch_model.yolo_nano.YOLONano(num_classes=num_classes, image_size=image_size)

class YoloNanoPaper(Profile):
    def get_model(self, num_classes, image_size):
        return pytorch_model.yolo_nano.YOLONano(num_classes=num_classes, image_size=image_size)

class YoloNanoSPP(Profile):
    def get_model(self, num_classes, image_size):
        return pytorch_model.yolo_nano_spp.YoloNanoSPP(num_classes=num_classes, image_size=image_size)