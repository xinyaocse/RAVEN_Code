import torch.nn as nn
import torch
from PIL import Image
import torch.nn.functional as Fn
import torchvision.transforms.functional as F
import pickle
import os
import numpy as np
from torchvision.transforms import transforms

from pooling import gempooling

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tf = transforms.Compose([transforms.Resize((224, 224)),
                         transforms.ToTensor()])  # 标准化
gem = gempooling.GeMPooling(2048, pool_size=3, init_norm=3.0).to(device)


def dcg(scores, k):
    """Compute the Discounted Cumulative Gain (DCG) at k.

    Args:
        scores (list or np.array): The relevance scores.
        k (int): The rank position to evaluate DCG at.

    Returns:
        float: The DCG value.
    """
    scores = np.asfarray(scores)[:k]
    if scores.size:
        return np.sum(scores / np.log2(np.arange(2, scores.size + 2)))
    return 0.0


def ndcg(scores, ideal_scores, k):
    """Compute the Normalized Discounted Cumulative Gain (NDCG) at k.

    Args:
        scores (list or np.array): The relevance scores.
        ideal_scores (list or np.array): The ideal relevance scores.
        k (int): The rank position to evaluate NDCG at.

    Returns:
        float: The NDCG value.
    """
    actual_dcg = dcg(scores, k)
    ideal_dcg = dcg(ideal_scores, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def load_image(imgpath):
    image = Image.open(imgpath)
    image = tf(image)
    image = image.unsqueeze(0)
    return image.cuda()


def gemfeature(tensor):
    gem_feature = gem(tensor)
    gem_feature = gem_feature.flatten()
    return gem_feature.cuda()


def guide_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)


def l1_loss(output, bicubic_image):
    loss_fn = torch.nn.L1Loss(reduction='mean')
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def load(name, net):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        input = input / 255.0
        mean = self.mean.reshape(1, 3, 1, 1).to(device)
        std = self.std.reshape(1, 3, 1, 1).to(device)
        return (input - mean) / std


def cifar_name(number):
    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return str(label_names[number])


def normal_r(output_r):
    r_max = torch.max(output_r)
    r_min = torch.min(output_r)
    r_mean = r_max - r_min
    output_r = (output_r - r_min) / r_mean
    return output_r


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


def imglist(path, mat):
    dirpath = []
    for parent, dirname, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(mat):
                dirpath.append(os.path.join(parent, filename))
    return dirpath


def testlist(path):
    dirpath = []
    for parent, dirname, filenames in os.walk(path):
        for filename in filenames:
            if filename.find("result") != -1:
                dirpath.append(os.path.join(parent, filename))

    return dirpath


def l_cal(img1, img2):
    noise = (img1 - img2).flatten(start_dim=0)
    l2 = torch.sum(torch.pow(torch.norm(noise, p=2, dim=0), 2))
    l_inf = torch.sum(torch.norm(noise, p=float('inf'), dim=0))
    return l2, l_inf


# 对输入的图像进行缩放填充操作
def input_diversity(input_tensor, ratio):
    # 放缩到1/2
    scale = int(224 * ratio)
    scaled_tensor = Fn.interpolate(input_tensor, size=(scale, scale), mode='bilinear', align_corners=False)
    # 填充图像到(1, 3, 224, 224)
    pad_top = int((224 - scale) / 2)
    pad_bottom = int((224 - scale) / 2)
    pad_left = int((224 - scale) / 2)
    pad_right = int((224 - scale) / 2)
    # 0是黑 1是白
    padded_tensor = Fn.pad(scaled_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    return padded_tensor


# 放大变换 但是会失真
def zoom_in_center(image_tensor, zoom_factor):
    # 获取图像的原始尺寸
    _, _, height, width = image_tensor.size()

    # 计算局部放大后的尺寸
    zoomed_width = int(width * zoom_factor)
    zoomed_height = int(height * zoom_factor)

    # 将图像缩放到局部放大后的尺寸
    zoomed_image_tensor = F.resize(image_tensor, (zoomed_height, zoomed_width))
    # print(zoomed_image_tensor.size())  # torch.Size([1, 3, 448, 448])
    # 计算粘贴的位置，使得放大后的图像处于原始图像的中心
    paste_x = int((zoomed_width - width) / 2)
    paste_y = int((zoomed_height - height) / 2)

    # 创建一个和原始图像大小一样的空白图像
    enlarged_image_tensor = torch.zeros_like(image_tensor)

    # 将放大后的图像粘贴到空白图像的中心位置
    enlarged_image_tensor = zoomed_image_tensor[:, :, paste_y:paste_y + height, paste_x:paste_x + width]

    return enlarged_image_tensor


# 图片平移
def translate_image(image_tensor, x_shift):
    shift_x = x_shift
    transform_matrix = torch.tensor([
        [1, 0, shift_x],
        [0, 1, 0]]).unsqueeze(0)  # 设B(batch size为1)
    grid = torch.nn.functional.affine_grid(transform_matrix,  # 旋转变换矩阵
                                           image_tensor.shape).to(device)  # 变换后的tensor的shape(与输入tensor相同)

    translated_tensor = torch.nn.functional.grid_sample(image_tensor,  # 输入image_tensor，shape为[B,C,W,H]
                                                        grid,  # 上一步输出的gird,shape为[B,C,W,H]
                                                        mode='nearest').to(device)  # 一些图像填充方法，这里我用的是最近邻
    # 输出output的shape为[B,C,W,H]
    return translated_tensor


# 打开cifar-10数据集文件目录
def unpickle(file):
    with open("../cifar-10-python/cifar-10-batches-py/" + file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
        # data_batch为字典，包含四个字典键：
        # batch_label
        # labels 标签
        # data  图片像素值
        # filenames
    return dic
