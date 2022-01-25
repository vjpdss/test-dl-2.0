# Transforms applied to the dataset in dataset.py
import copy
import torch
from torchvision import transforms


class ResizeImgBbox:
    def __init__(self, new_size):
        self.new_size = new_size
        self.resize = transforms.Resize(new_size)  # height, width

    def __call__(self, sample):
        # Resize images and bounding boxes
        sample_resized = copy.copy(sample)
        image, target = sample['img'], sample['target']
        w, h = image.size
        image_resized = self.resize(image)

        target_resized = dict()
        target_resized['bboxes'] = []
        scale_x = self.new_size[1] / w
        scale_y = self.new_size[0] / h
        for bbox in target['bboxes']:
            # bbox format: x, y, width, height
            resized_bbox = [bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y]
            target_resized['bboxes'].append(resized_bbox)

        target_resized['labels'] = target['labels']
        sample_resized['img'] = image_resized
        sample_resized['target'] = target_resized

        return sample_resized


class ToTensor:
    # Converts image from the dictionary to a tensor that can be fed to the model.
    def __init__(self):
        self.ToTensor = transforms.ToTensor()

    def __call__(self, sample):
        image, target = sample['img'], sample['target']
        target['labels'] = torch.Tensor(target['labels']).type(dtype=torch.uint8)
        target['bboxes'] = torch.Tensor(target['bboxes']).type(dtype=torch.float32)

        return {'img': self.ToTensor(image),
                'target': target}


class NormalizeInverse(transforms.Normalize):
    # Taken from https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/7
    """
    Undoes the normalization and returns the reconstructed images to the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
