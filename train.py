import time

import torch
from torch.utils.data import DataLoader
from torch import nn
from model_mae import MAE, Vit
from dataset import COCODataSet

num_epoch = 1000000  # 迭代次数
num_batch = 1  # batch_size
num_patch = 14  # 图片几等分
image_size = 224  # 图片尺寸
mask_ratio = 0.75  # 遮罩率
learn_rate = 0.001  # 学习率
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 设备选择
dataset = COCODataSet('data', image_size)  # 数据集
dataset_loader = DataLoader(dataset, num_batch, True, drop_last=True)  # 数据加载器
vit = Vit(num_patch)  # vit
mae = MAE(vit, num_patch=num_patch, image_size=image_size, mask_ratio=mask_ratio).to(device)  # MAE

optimizer = torch.optim.Adam(mae.parameters(), lr=learn_rate)  # 优化器
loss_function = nn.MSELoss()  # 损失函数
if __name__ == '__main__':
    for epoch in range(num_epoch):
        loss_sum = 0
        for image_tensor in dataset_loader:
            image_tensor = image_tensor.to(device)
            pred_pixel_values, pred_mask_pixel_values, mask_patches = mae(image_tensor)[:3]
            loss = loss_function(pred_mask_pixel_values, mask_patches)

            """训练三件套"""
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            time.sleep(0.1)

        print(f'Epoch: {epoch}; Loss: {loss_sum / len(dataset_loader)}')
        if epoch % 100 == 0:
            torch.save(mae.state_dict(), 'mae.pt')
