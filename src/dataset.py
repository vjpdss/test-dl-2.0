import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import my_transforms


class MyDataset(Dataset):
    def __init__(self, data_path, labels, transform=None):
        # Data path
        self.data_path = data_path

        # Available labels for object detection
        self.labels_str = ['__background__', 'ball', 'player', 'referee', 'person']
        self.labels_map_to_player = ['referee']  # If not defined in labels map this label to 'player'.
        self.labels_to_use = labels
        self.labels = dict()

        # Set transform method
        self.transform = transform

        # Labels to num
        self._init_labels_dict_()

        # List all the images and xml available
        self.images = None
        self.data = None
        self._list_imgs_xmls_()

        # In order to compute mean and std we don't read XML files
        self.no_xml = False

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Read image
        img = Image.open(self.images[index], 'r')
        img = img.convert('RGB')

        # Read objects in the scene
        xml = self.data[index]
        objects = self.parse_info_from_xml(xml)
        labels = objects[0]
        bboxes = objects[1]
        target = {
            'labels': labels,
            'bboxes': bboxes
        }

        sample = {
            'img': img,
            'target': target
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _list_imgs_xmls_(self):
        # List all the images and xml available
        self.images = [img for img in self.data_path.iterdir() if img.suffix == '.jpg' or img.suffix == '.png']
        self.data = [xml for xml in self.data_path.iterdir() if xml.suffix == '.xml']
        # Order the data so it matches
        self.images = sorted(self.images)
        self.data = sorted(self.data)

    def _init_labels_dict_(self):
        for l_num, l_str in enumerate(self.labels_to_use):
            self.labels[l_str] = l_num + 1  # 0 is background

    def parse_info_from_xml(self, xml_path):
        # Return: labels and their corresponding bounding boxes
        print('TODO: Parse the information from XMLs')
        # hint: return labels, bboxes

    def label_str_to_num(self, l_str):
        # Pass label string to number
        l_num = -1
        if l_str in self.labels_to_use:
            l_num = self.labels[l_str]
        elif l_str in self.labels_map_to_player:
            l_num = self.labels['player']

        return l_num

    def collate_fn(self, batch):
        # Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b['img'])
            boxes.append(b['target']['bboxes'])
            labels.append(b['target']['labels'])

        images = torch.stack(images, dim=0)

        return images, boxes, labels  # tensor (N, 3, 300, 300), 3 lists of N tensors each


def compute_mean_std_dataset():
    # Compute dataset mean and std
    root_path = Path(args.root_path)
    split_path = root_path / 'train'
    transform = transforms.Compose([
        my_transforms.ResizeImgBbox((544, 960)),
        my_transforms.ToTensor()
    ])
    my_dataset = MyDataset(split_path, ['ball', 'player', 'referee', 'person'], transform=transform)
    my_dataloader = DataLoader(my_dataset, batch_size=10, shuffle=True, collate_fn=my_dataset.collate_fn, num_workers=0)

    # Taken from https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/6
    mean = 0.
    std = 0.
    nb_samples = 0.

    for i_batch, sample_batch in enumerate(my_dataloader):
        imgs_batch = sample_batch[0]
        batch_samples = imgs_batch.size(0)
        imgs_batch = imgs_batch.view(batch_samples, imgs_batch.size(1), -1)

        mean += imgs_batch.mean(2).sum(0)
        std += imgs_batch.std(2).sum(0)
        nb_samples += batch_samples

        if i_batch % 10 == 0:
            print('mean: {}, std: {}, it: {} / {}'.format(mean / nb_samples, std / nb_samples, i_batch * 10, len(my_dataset)))

        if i_batch > 100:
            # Take 1000 samples (to speed up)
            break

    mean /= nb_samples
    std /= nb_samples

    print("mean: " + str(mean))
    print("std: " + str(std))


def test_dataloader():
    # Function to test the dataloader
    root_path = Path(args.root_path)
    split_path = root_path / 'train'
    my_dataset = MyDataset(split_path, ['ball', 'player', 'referee', 'person'])

    for i in range(10):
        idx_rdn = np.random.randint(0, len(my_dataset) - 1, 1)[0]
        print(idx_rdn)
        sample = my_dataset[idx_rdn]
        img, target = sample['img'], sample['target']


if __name__ == '__main__':
    # E.g. data/match1
    parser = argparse.ArgumentParser(description='Dataset generator and dataloader.')
    parser.add_argument('data_path', type=str, help='Data path.')

    args = parser.parse_args()
    test_dataloader()

    # To compute mean and std of the dataset for image normalization
    # compute_mean_std_dataset()
