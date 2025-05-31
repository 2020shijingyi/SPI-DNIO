import torch
import numpy as np
from PIL import Image
from einops import rearrange
import os

def measurement_encoding(T,patterns):
    '''

    :param T: b,c,h,w
    :param patterns: b,c,h,w,N
    :return: measurements
    '''
    b, c, _, _ = T.size()
    T_repet = T.unsqueeze(-1).repeat(1, 1, 1, 1, patterns.shape[4])
    product = T_repet * patterns

    measurements = torch.sum(torch.sum(product, dim=2), dim=2)
    measurements = measurements.reshape(b, c, -1)

    return measurements

def DGI_reconstruction(measurements, patterns):
    """
    :param measurements: b,c,N
    :param patterns: b,c,h,w,N
    """
    # 初始化变量
    SI_aver = 0
    B_aver = 0
    R_aver = 0
    RI_aver = 0
    iter = 0

    # 循环处理每个图案
    for i in range(patterns.shape[4]):
        # 获取当前图案
        Br = measurements[:, :, i].unsqueeze(-1).unsqueeze(-1)
        pattern = patterns[:, :, :, :, i]

        # 更新迭代次数
        iter += 1

        # Differential ghost imaging (DGI)
        SI_aver = (SI_aver * (iter - 1) + pattern * Br) / iter
        B_aver = (B_aver * (iter - 1) + Br) / iter
        R_aver = (R_aver * (iter - 1) + torch.sum(pattern)) / iter
        RI_aver = (RI_aver * (iter - 1) + torch.sum(pattern) * pattern) / iter

    # 计算重建图像
    DGI = SI_aver - B_aver / R_aver * RI_aver
    return DGI

def Normalized_std(float_data):

    batch_size = float_data.shape[0]
    batches = []
    for i in range(batch_size):
        batch = float_data[i]
        batch = (batch - batch.min()) / (batch.max() - batch.min())
        batches.append(batch)
    batchs_all = torch.stack(batches)
    float_data =batchs_all

    return float_data

def Normalized_max(float_data):
    # 最大值归一化
    batch_size = float_data.shape[0]
    batches = []
    for i in range(batch_size):
        batch = float_data[i]
        batch = (batch) / (batch.max())
        batches.append(batch)
    batchs_all = torch.stack(batches)
    float_data =batchs_all

    return float_data


class EarlyStopping:
    def __init__(self, patience=30):
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float('inf')  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded

        return stop

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_data(path):
    '''

    :param path: data path
    :return: data (torch) b,c,h,w 0-1
    '''
    image = Image.open(path).convert('L')
    image = np.stack([image] * 1, axis=-1)
    image = np.expand_dims(image, axis=0)
    image = (image[..., ::-1] / 255.0).astype(np.float32)
    data = image[..., ::-1].astype(np.float32)
    data = torch.tensor(data)
    data = rearrange(data, "b h w c -> b c h w").contiguous().float().clamp(0,1)

    return data

def add_noise_to_mea(mea,dSNR):
    '''

    :param mea: b,c,N
    :param dSNR:
    :return: mea_noise
    '''
    noise_level = torch.mean(mea) / 10 ** (dSNR / 10)
    noise = torch.randn_like(mea) * noise_level
    mea_noise = mea + noise

    return mea_noise

def total_variation(image):
    # 计算水平方向的差异
    h_diff = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])
    # 计算垂直方向的差异
    v_diff = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])
    # 计算总变差
    tv = torch.sum(h_diff) + torch.sum(v_diff)
    return tv

def calc_psnr(img1, img2):
    '''
    one channel and Normalize
    :param img1:
    :param img2:
    :return:
    '''
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

def save_img(data, name, filepath='result/'):
    '''
    保存 4D 图像张量为 PNG 图像，支持单通道（灰度）和三通道（RGB）。

    参数：
    - data: torch.Tensor，形状为 (B, C, H, W)，值应在 [0, 1] 范围
    - name: 保存图像的文件名（不含扩展名）
    - filepath: 图像保存的文件夹路径
    '''
    os.makedirs(filepath, exist_ok=True)  # 自动创建文件夹
    image_tensor = data[0]  # (C, H, W)
    image_tensor = image_tensor.detach().cpu()  # 确保在 CPU 并移除计算图

    # 将像素缩放到 [0, 255]
    image_np = (image_tensor * 255).clamp(0, 255).to(torch.uint8).numpy()

    if image_np.shape[0] == 1:  # 单通道灰度图
        image_np = image_np[0]  # shape: (H, W)
        image = Image.fromarray(image_np, mode='L')  # 灰度图像
    elif image_np.shape[0] == 3:  # 三通道 RGB 图
        image_np = np.transpose(image_np, (1, 2, 0))  # shape: (H, W, C)
        image = Image.fromarray(image_np, mode='RGB')
    else:
        raise ValueError(f"Unsupported number of channels: {image_np.shape[0]}")

    image.save(os.path.join(filepath, f'{name}.png'))