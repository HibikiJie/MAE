import torch
from torch.utils.data import DataLoader
from torch import nn
import torchvision
from model_mae import MAE, Vit
from dataset import COCODataSet


num_epoch = 1000
num_batch = 1
num_patch = 14
image_size = 224
mask_ratio = 0.75
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset = COCODataSet('data', image_size, False)
dataset_loader = DataLoader(dataset, num_batch, True, drop_last=True)
vit = Vit(num_patch)
to_PIL_image = torchvision.transforms.ToPILImage()
mae = MAE(vit, num_patch=num_patch, image_size=image_size, mask_ratio=mask_ratio).to(device)
mae.load_state_dict(torch.load('mae.pt'))
mae.eval()
if __name__ == '__main__':
        for image_tensor in dataset_loader:
            image_tensor = image_tensor.to(device)
            recons_img, patches_to_img = mae.predict(image_tensor)
            mse = ((image_tensor - recons_img)**2).mean()
            original_image = to_PIL_image(image_tensor[0])
            recons_img = to_PIL_image(recons_img[0])
            masked_img = to_PIL_image(patches_to_img[0])

            ##plt 同时显示多幅图像
            import matplotlib.pyplot as plt
            print('MSE: ',mse)
            plt.figure()
            plt.subplot(1, 3, 1)
            plt.imshow(original_image)
            plt.subplot(1, 3, 2)
            plt.imshow(recons_img)
            plt.subplot(1, 3, 3)
            plt.imshow(masked_img)
            plt.show()
