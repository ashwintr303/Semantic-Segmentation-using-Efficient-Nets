import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
from torch.autograd import Variable

# define the network archtecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # encoder block
        self.encoder = ptcv_get_model("efficientnet_b0b", pretrained=True).features
        #self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        for param in self.encoder.parameters():
            param.requires_grad = False

        # decoder block
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 1280, (12, 39), stride=1, padding=0),  ## (1280,12,39)
            nn.ReLU(),
            nn.ConvTranspose2d(1280, 672, (1, 1), stride=1, padding=0),  ## (672,12,39)
            nn.ReLU(),
            nn.ConvTranspose2d(672, 672, (12, 39), stride=1, padding=0),  ## (672,23,77)
            nn.ReLU(),
            nn.ConvTranspose2d(672, 480, (1, 1), stride=1, padding=0),  ## (480,23,77)
            nn.ReLU(),
            nn.ConvTranspose2d(480, 240, (1, 1), stride=1, padding=0),  ## (240,23,77)
            nn.ReLU(),
            nn.ConvTranspose2d(240, 240, (24, 79), stride=1, padding=0),  ## (240,46,155)
            nn.ReLU(),
            nn.ConvTranspose2d(240, 144, (1, 1), stride=1, padding=0),  ## (144,46,155)
            nn.ReLU(),
            nn.ConvTranspose2d(144, 144, (46,155), stride=1, padding=0),  ## (144,92,310)
            nn.ReLU(),
            nn.ConvTranspose2d(144, 96, (93, 310), stride=1, padding=0),  ## (96,185,620)
            nn.ReLU(),
            nn.ConvTranspose2d(96, 32, (46, 155), stride=1, padding=0),  ## (32,185,620)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, (185, 620), stride=1, padding=0),  ## (32,370,1240)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, (185, 620), stride=1, padding=0),  ## (3,370,1240)
            nn.ReLU()
        )

    def forward(self, x):
        out = self.encoder(x)
        print('first', out.size())
        out = self.decoder(out)
        print('second',out.size())
        return out

model = Net()