from torch import nn
import torch


class MAE(nn.Module):
    def __init__(
            self,
            encoder,  # 编码器
            num_patch=14,  # 划分的个数
            image_size=112,
            mask_ratio=0.75,  # 掩码比率
    ):
        super(MAE, self).__init__()
        self.num_patch = num_patch
        """编码器"""
        self.encoder = encoder

        """解码器"""
        self.decoder = nn.Sequential(
            nn.TransformerEncoderLayer(512, 8, dropout=0.2, batch_first=True),
        )

        """图像转换至embedding的线性层"""
        self.path_length = int(image_size / num_patch)
        out_features = int(self.path_length ** 2 * 3)
        self.embedding_to_token = nn.Linear(out_features, 512)

        """头部，预测像素值"""
        self.head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, out_features=out_features)
        )

        """mask部分的token"""
        self.mask_embedding = nn.Parameter(torch.randn(1, 1, 512))

        """位置编码"""
        self.position_embedding = nn.Parameter(torch.randn(1, num_patch ** 2, 512))

        self.num_masked = int(mask_ratio * num_patch * num_patch)  # 掩码率

    def forward(self, x):
        """

        :param x:
        :return:
        """

        """image划分patch,并变换形状至(N, S, V)"""
        n, c, h, w = x.shape
        p = self.num_patch
        w = h = w // p
        x = x.reshape(n, c, p, h, p, w)
        x = torch.einsum('ncphqw->npqhwc', x)
        x = x.reshape(n, p **2, h*w * 3)
        # print(n, c, h, p, w, p)
        # print(x.shape)
        """生成掩码"""
        shuffle_indication = torch.rand(n, p ** 2).argsort()  # 生成随机数
        batch_index = torch.arange(n).unsqueeze(-1)
        mask_index, unmask_index = shuffle_indication[:, :self.num_masked], shuffle_indication[:, self.num_masked:]
        mask_patches, unmask_patches = x[batch_index, mask_index], x[batch_index, unmask_index]

        """为遮盖的部分通过vit编码"""
        unmask_tokens = self.embedding_to_token(unmask_patches)
        encoded_tokens = self.encoder(unmask_tokens, index=unmask_index)

        """获取掩码的token"""
        mask_tokens = self.mask_embedding.repeat(n, self.num_masked, 1)

        """恢复次序"""
        concat_tokens = torch.cat([mask_tokens, encoded_tokens], dim=1)
        decoder_input_tokens = torch.empty_like(concat_tokens, device=concat_tokens.device)
        decoder_input_tokens[batch_index, shuffle_indication] = concat_tokens
        decoder_input_tokens += self.position_embedding  # 添加位置编码


        """解码"""
        decoded_tokens = self.decoder(decoder_input_tokens)

        pred_pixel_values = self.head(decoded_tokens)  # 预测遮盖的图像像素值
        pred_mask_pixel_values = pred_pixel_values[batch_index, mask_index]  # 取出遮盖的预测像素值
        return pred_pixel_values, pred_mask_pixel_values, mask_patches, x, batch_index, mask_index

    @torch.no_grad()
    def predict(self, x):
        """

        :param x:
        :return:
        """
        self.eval()
        n, c, h, w = x.shape
        p = self.num_patch
        h = w = h // p
        pred_pixel_values, pred_mask_pixel_values, mask_patches, patches, batch_index, mask_index = self.forward(x)
        # mask_patches = torch.zeros_like(mask_patches, device=mask_patches.device)
        patches[batch_index, mask_index] = pred_mask_pixel_values
        recons_img = patches.reshape(n, p, p, h, w, c)
        recons_img = torch.einsum('npqhwc->ncphqw', recons_img)
        recons_img = recons_img.reshape(n, c, h*p, w*p)

        mask_patches = torch.zeros_like(mask_patches, device=mask_patches.device)
        patches[batch_index, mask_index] = mask_patches
        patches_to_img = patches.reshape(n, p, p, h, w, c)
        patches_to_img = torch.einsum('npqhwc->ncphqw', patches_to_img)
        patches_to_img = patches_to_img.reshape(n, c, h*p, w*p)
        return recons_img, patches_to_img


class Vit(nn.Module):
    def __init__(
            self,
            num_path=14,
    ):
        super(Vit, self).__init__()
        self.layer = nn.Sequential(
            nn.TransformerEncoderLayer(512, 8, 2048, dropout=0.2, batch_first=True),
            nn.TransformerEncoderLayer(512, 8, 2048, dropout=0.2, batch_first=True),
            nn.TransformerEncoderLayer(512, 8, 2048, dropout=0.2, batch_first=True),
            nn.TransformerEncoderLayer(512, 8, 2048, dropout=0.2, batch_first=True),
            nn.TransformerEncoderLayer(512, 8, 2048, dropout=0.2, batch_first=True),
            nn.TransformerEncoderLayer(512, 8, 2048, dropout=0.2, batch_first=True),
            nn.TransformerEncoderLayer(512, 8, 2048, dropout=0.2, batch_first=True),
        )
        num_embedding = int(num_path ** 2)
        self.position_embedding = nn.Embedding(num_embedding, 512)

    def forward(self, x, index=None):
        if index is not None:
            x = x + self.position_embedding(index)
        return self.layer(x)


if __name__ == '__main__':
    pass
    vit = Vit()
    mae = MAE(vit,image_size=224)
    x = torch.randn(2, 3, 224, 224)
    mae(x)
