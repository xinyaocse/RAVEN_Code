import os
import glob

from torch.autograd import Variable
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torchvision
import warnings
from torchvision.utils import save_image
from args import get_args_parser
from util.utils import *
from model.model import *

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#####################
# Model initialize: #
#####################
args = get_args_parser()
tf = transforms.Compose([transforms.Resize((224, 224)),
                         transforms.ToTensor()])  # 标准化
checkpoint = torch.load("./checkpoints/CIFAR10/binary/resnet50_trained_checkpoint.pth")
state_dict = checkpoint['model_state_dict']
model = torchvision.models.resnet50()
model.fc = nn.Linear(2048, 2)
model.load_state_dict(state_dict)
model.eval().to(device)
model_feature = torch.nn.Sequential(*list(model.children())[:-2])
model_feature.eval().to(device)

data_dir = ''


file_paths = sorted(glob.glob(os.path.join(data_dir, '*')))

# select 10 target image
paths = file_paths[:10]
# 数据库检索
train_data = torchvision.datasets.ImageFolder('./CIFAR10/train', transform=tf)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False,
                                           pin_memory=True, drop_last=True)

# such an query image
cover = load_image('./CIFAR10/train/dog/dog88.jpg').to(device)
# gem = gempooling.GeMPooling(2048, pool_size=3, init_norm=3.0).to(device)
# mac = nn.MaxPool2d(kernel_size=2, stride=2)
cover_feature = model_feature(cover)
# 变量
scores = []

# adversarial radius
xi = 8 / 255.0
# 2、往中心区域
if __name__ == '__main__':
    for i in range(10):
        # original target image
        target_image = load_image(paths[i]).to(device)
        # target feature
        target_feature = model_feature(target_image).to(device)
        perturbation = (torch.rand(1, 3, 224, 224).to(device) - 0.5) * 2 * xi
        perturbation = Variable(perturbation, requires_grad=True)
        optim = torch.optim.Adam([perturbation], lr=0.01)
        best_num = 0.0
        for i_epoch in range(50):
            scores = []
            for j, data in enumerate(train_loader):
                #
                image, label = data[0].to(device), data[1].to(device)
                power_feature = model_feature(target_image + perturbation).to(device)
                feature2 = model_feature(image).to(device)
                score = guide_loss(power_feature, feature2).to(device)
                scores.append([label, score.item()])
            scores.sort(key=lambda x: x[1])
            power_vector = torch.zeros(1, 10).to(device)
            target_vector = torch.ones(1, 10).to(device)
            # retrieval list of top-10
            for index in range(10):
                labels, _ = scores[index]
                if labels == 1:
                    power_vector[0, index] = 1
            center_fea_loss = guide_loss(target_feature, power_feature).to(device)
            center_list_loss = (target_vector.sum() - power_vector.sum()).to(device)
            # far away from query sample
            cover_fea_loss = guide_loss(power_feature, cover_feature).to(device)
            total_loss = center_list_loss + 10 * (center_fea_loss - cover_fea_loss)
            optim.zero_grad()
            perturbation.data = torch.clamp(perturbation.data, -xi, xi)
            total_loss.backward(retain_graph=True)
            optim.step()

            if power_vector.sum() > best_num:
                # save the best result
                best_num = power_vector.sum()
                best_result = target_image + perturbation.data
                save_image(best_result, args.outputpath + str(i) + ".png")
            print(f"本轮迭代检索出{power_vector.sum()},最佳检索出{best_num}")
            if best_num == 0 and i_epoch == 100:
                # iteration end
                inverse = target_image + perturbation.data
                save_image(inverse, args.outputpath + str(i) + ".png")
            print(
                'Epoch [{}/{}], total_loss: {:.4f},center_fea_loss:{:.4f},cover_fea_loss:{:.4f}'.format(
                i_epoch + 1, 50, total_loss.item(), center_fea_loss.item(), cover_fea_loss.item()))