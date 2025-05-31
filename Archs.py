import numpy as np
from Util import measurement_encoding,DGI_reconstruction,Normalized_std
import torch
import torch.nn as nn

def truncated_normal_(tensor, mean=0.0, std=0.001):
    '''
    :param tensor:
    :param mean:
    :param std: BEST 0.001
    :return:
    '''
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
class ResBlock(nn.Module):

    def __init__(
        self,
        channels,
        channels_hind,
        out_channels=None,

    ):
        super().__init__()
        self.out_channels = out_channels or channels
        self.layer1 = nn.Sequential(*[
            nn.Conv2d(in_channels=channels,out_channels=channels_hind,kernel_size=5,padding='same'),
            nn.BatchNorm2d(channels_hind),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels_hind, out_channels=channels_hind, kernel_size=5, padding='same'),
            nn.BatchNorm2d(channels_hind),
            nn.LeakyReLU(),

        ])

        self.layer_L = nn.Sequential(*[
            nn.Conv2d(in_channels=channels_hind//2,out_channels=channels_hind,kernel_size=5,stride=2,padding=2),
            nn.BatchNorm2d(channels_hind),
            nn.LeakyReLU()
        ])
        self.layer_R = nn.Sequential(*[
            nn.Conv2d(in_channels=channels_hind // 2, out_channels=channels_hind, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(channels_hind),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=channels_hind, out_channels=channels_hind, kernel_size=5,padding='same'),
            nn.BatchNorm2d(channels_hind),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=channels_hind, out_channels=channels_hind, kernel_size=5,padding='same'),
            nn.BatchNorm2d(channels_hind),
            nn.LeakyReLU()
        ])

        self.layer_cat = nn.Sequential(*[
            nn.Conv2d(in_channels=2*channels_hind,out_channels=channels_hind,kernel_size=1,padding='same'),
            nn.BatchNorm2d(channels_hind),
            nn.LeakyReLU()
        ])
        self.layer_up = nn.Sequential(*[
            nn.Conv2d(in_channels=channels_hind,out_channels=4*channels_hind,kernel_size=3,padding='same'),
            nn.PixelShuffle(upscale_factor=2),
            nn.Conv2d(in_channels=channels_hind,out_channels=channels_hind,kernel_size=3,padding='same'),
            nn.BatchNorm2d(channels_hind)
        ])

        self.skip = nn.Sequential(*[
            nn.Conv2d(in_channels=channels,out_channels=self.out_channels,kernel_size=1,padding='same'),
            nn.BatchNorm2d(self.out_channels)
        ])
        self.skip2 = nn.Sequential(*[
            nn.Conv2d(in_channels=channels_hind,out_channels=self.out_channels,kernel_size=1,padding='same'),
            nn.BatchNorm2d(self.out_channels)
        ])



    def forward(self, x):

        o1 = self.layer1(x)
        o2_L,o2_R = torch.chunk(o1,2,dim = 1)
        o2_L = self.layer_L(o2_L)
        o2_R = self.layer_R(o2_R)
        o3 = self.layer_cat(torch.cat([o2_L,o2_R],dim=1))
        o4 = self.layer_up(o3)+o1
        out = self.skip(x)+self.skip2(o4)

        return out

class ResModel(nn.Module):


    def __init__(
            self,
            in_channels,
            out_channels,
            em_channels,
            res_num
    ):
        super().__init__()
        self.layer1 = nn.Sequential(*[
            nn.Conv2d(in_channels=in_channels,out_channels=em_channels,kernel_size=3,padding='same'),
            nn.BatchNorm2d(em_channels),
            nn.ReLU(),
        ])
        self.blocks_input = nn.Sequential(*[
            ResBlock(channels=em_channels,channels_hind=em_channels,out_channels=em_channels) for i in range(res_num)
        ])

        self.layer_out = nn.Sequential(*[
            nn.Conv2d(in_channels=em_channels,out_channels=out_channels,kernel_size=3,padding='same'),
            nn.BatchNorm2d(out_channels),
        ])
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                truncated_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        o1 = self.layer1(x)
        o2 = self.blocks_input(o1)
        out = self.layer_out(o2)+x

        return out

class Pattern_ini(nn.Module):
    def __init__(self,output_dim=(64, 64, 1), input_dim=(64, 64, 1),compression = 0.25):

        super(Pattern_ini, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.compression = compression

        M = round((self.input_dim[0] * self.input_dim[1] * self.input_dim[2]) * self.compression)
        H_init = np.random.normal(0, 1, (1, self.input_dim[0], self.input_dim[0], M)) / np.sqrt(
            self.input_dim[0] * self.input_dim[0])

        H_init = torch.tensor(np.expand_dims(H_init,axis=0),dtype=torch.float)


        self.H = nn.Parameter(H_init,requires_grad=True)


class Model(nn.Module):
    def __init__(self,res_num = 14,compression = 0.25):
        super(Model, self).__init__()
        # RGI-RNet-s res-num 9；RGI-RNet res-num 14
        self.re_model = ResModel(in_channels=1,out_channels=1,res_num=res_num,em_channels=64)
        self.H = Pattern_ini(compression=compression).H
    def forward(self, x):
        mea =measurement_encoding(x,self.H.to(x.device))
        DGI = DGI_reconstruction(mea,patterns=self.H)
        DGI = Normalized_std(DGI)
        DGI_rec = self.re_model(DGI)
        return DGI_rec,DGI,self.H
    def forward_(self,x):
        DGI_re = self.re_model(x)
        return DGI_re


