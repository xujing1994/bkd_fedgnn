from Common.Model.LeNet import LeNet
from Common.Model.ResNet import ResNet, BasicBlock
from Common.Utils.data_loader import load_data_mnist
import torch
import torchvision.models as models



PATH = './Model/LeNet'
#PATH = './Model/ResNet18'
#model = models.resnet18()
model = LeNet()
#model = ResNet(BasicBlock, [3,3,3])
torch.save(model.state_dict(), PATH)
model_load = LeNet()
#model_load = models.resnet18()
#model_load = ResNet(BasicBlock, [3,3,3])
model_load.load_state_dict(torch.load(PATH))
print(model_load)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



for name in model.state_dict():
    print(name)
layerslen = []
for p in model.parameters():
    if p.requires_grad:
     layerslen.append(p.numel())
layerslen = [sum(layerslen[:i + 1]) for i in range(len(layerslen))]
layerslen = layerslen[:-1]
print(layerslen)
print(len(layerslen))
print('parameters_count:',count_parameters(model))
