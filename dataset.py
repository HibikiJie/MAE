import os

import cv2
from torch.utils.data import Dataset
import torchvision
from PIL import Image
import numpy


class COCODataSet(Dataset):

    def __init__(self, data_path, image_size, is_train=True):
        super(COCODataSet, self).__init__()
        self.data_set = []
        for file_name in os.listdir(data_path):
                self.data_set.append(f'{data_path}/{file_name}')
        if is_train:
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(size=(image_size, image_size)),
                    # torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                    # torchvision.transforms.RandomApply([color_jitter], p=0.8),
                    # torchvision.transforms.RandomGrayscale(p=0.2),
                    torchvision.transforms.ToTensor(),
                    # torchvision.transforms.Normalize(mean=[0.471, 0.448, 0.408], std=[0.234, 0.239, 0.242])
                ]
            )
        else:
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(size=(image_size, image_size)),
                    torchvision.transforms.ToTensor(),
                ]
            )

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, item):
        path = self.data_set[item]
        image = cv2.imread(path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # cv2.imshow('a', image)
        # cv2.waitKey()
        image = Image.fromarray(image)
        # image = image.convert("BGR")
        # image.show()
        image = self.transform(image)
        return image


if __name__ == '__main__':
    dataset = COCODataSet('data', 112)
    print(dataset[0].shape)