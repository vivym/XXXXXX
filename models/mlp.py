import torch.nn as nn
import torchvision.transforms as transforms
__all__=['MLP']


class MLPBase(nn.Module):
    def __init__(self, num_classes=0, in_dim=1, layers=2, hidden=7):
        super(MLPBase, self).__init__()

        out_layer_list = [hidden for i in range(layers)]
        if num_classes == 0:
            out_layer_list.append(1) #for regression
        else:
            out_layer_list.append(num_classes)
        
        in_layer_list = [hidden for i in range(layers)]
        in_layer_list.insert(0, in_dim)

        layers = []
        for input, output in zip(in_layer_list, out_layer_list):
            layers.append(nn.Linear(input, output))
            #add relu activations
            layers.append(nn.ReLU())
        layers.pop() #remove final relu layer

        self.model = nn.Sequential(*layers)
        print(self.model)
        
    def forward(self, x):
        return self.model(x)

class MLP:
    base = MLPBase
    args = list()
    kwargs = {}
    transform_train = transforms.ToTensor()
    transform_test = transforms.ToTensor()