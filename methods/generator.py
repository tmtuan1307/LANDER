import torch
import torch.nn as nn
import numpy as np


class NLGenerator(nn.Module):
    def __init__(self, ngf=64, img_size=32, nc=3, nl=100, label_emb=None, le_emb_size=256, le_size=512, sbz=200):
        super(NLGenerator, self).__init__()
        self.params = (ngf, img_size, nc, nl, label_emb, le_emb_size, le_size, sbz)
        self.le_emb_size = le_emb_size
        self.label_emb = torch.nn.Parameter(label_emb, requires_grad=False)
        self.init_size = img_size // 4
        self.le_size = le_size
        self.nl = nl
        self.sbz = sbz
        self.nle = int(np.ceil(self.sbz / self.nl))
        self.n1 = nn.BatchNorm1d(le_size)
        self.sig1 = nn.Sigmoid()
        self.le1 = nn.ModuleList([nn.Linear(le_size, le_emb_size) for i in range(self.nle)])
        self.l1 = nn.Sequential(nn.Linear(le_emb_size, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, targets=None):
        le = self.label_emb[targets]
        le = self.n1(le)
        v = None
        for i in range(self.nle):
            if (i + 1) * self.nl > le.shape[0]:
                sle = le[i * self.nl:]
            else:
                sle = le[i * self.nl:(i + 1) * self.nl]
            sv = self.le1[i](sle)
            if v is None:
                v = sv
            else:
                v = torch.cat((v, sv))

        out = self.l1(v)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    def re_init_le(self):
        for i in range(self.nle):
            nn.init.normal_(self.le1[i].weight, mean=0, std=1)
            nn.init.constant_(self.le1[i].bias, 0)

    def reinit(self):
        return NLGenerator(self.params[0], self.params[1], self.params[2], self.params[3], self.params[4],
                           self.params[5], self.params[6], self.params[7]).cuda()


class NLGenerator_IN(nn.Module):
    def __init__(self, ngf=64, img_size=224, nc=3, nl=100, label_emb=None, le_emb_size=256, le_size=512, sbz=200):
        super(NLGenerator_IN, self).__init__()
        self.params = (ngf, img_size, nc, nl, label_emb, le_emb_size, le_size, sbz)
        self.le_emb_size = le_emb_size
        self.label_emb = torch.nn.Parameter(label_emb, requires_grad=False)
        self.init_size = img_size // 16
        self.le_size = le_size
        self.nl = nl
        self.nle = int(np.ceil(sbz/nl))
        self.sbz = sbz

        self.n1 = nn.BatchNorm1d(le_size)
        self.sig1 = nn.Sigmoid()
        self.le1 = nn.ModuleList([nn.Linear(le_size, le_emb_size) for i in range(self.nle)])
        self.l1 = nn.Sequential(nn.Linear(le_emb_size, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            #nn.Conv2d(nz, ngf, 3, stride=1, padding=1, bias=False),
            #nn.BatchNorm2d(ngf),
            #nn.LeakyReLU(0.2, inplace=True),
            # 7x7

            #nn.Upsample(scale_factor=2),
            nn.Conv2d(2*ngf, 2*ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2*ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 14x14

            nn.Upsample(scale_factor=2),
            nn.Conv2d(2*ngf, 2*ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2*ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 28x28

            nn.Upsample(scale_factor=2),
            nn.Conv2d(2*ngf, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 56x56

            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 112 x 112

            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # 224 x 224

            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def re_init_le(self):
        for i in range(self.nle):
            nn.init.normal_(self.le1[i].weight, mean=0, std=1)
            nn.init.constant_(self.le1[i].bias, 0)

    def forward(self, targets=None):
        le = self.label_emb[targets]
        # le = self.sig1(le)
        le = self.n1(le)
        v = None
        for i in range(self.nle):
            if (i+1)*self.nl > le.shape[0]:
                sle = le[i*self.nl:]
            else:
                sle = le[i*self.nl:(i+1)*self.nl]
            sv = self.le1[i](sle)
            if v is None:
                v = sv
            else:
                v = torch.cat((v, sv))

        out = self.l1(v)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    def reinit(self):
        return NLGenerator_IN(self.params[0], self.params[1], self.params[2], self.params[3], self.params[4],
                               self.params[5], self.params[6], self.params[7]).cuda()


