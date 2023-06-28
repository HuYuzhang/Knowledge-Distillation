import torch
import torch.nn as nn
from thop import profile

class TDModel_3f(nn.Module):

    def __init__(self, opts, nc_out, nc_ch):
        super(TDModel_3f, self).__init__()

        self.epoch = 0

        use_bias = True

        self.conv1 = nn.Conv3d(3, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)#nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[1, 1, 1])
        self.conv5 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv7 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)#nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[1, 1, 1])
        self.conv8 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv10 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)#nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[1, 1, 1])
        self.conv11 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv13 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)#nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[1, 1, 1])
        self.conv14 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv15 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv16 = nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])
        self.conv17 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv18 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv19 = nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])
        self.conv20 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv21 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv22 = nn.Conv3d(nc_ch, nc_out, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, X, collect_feat=False):

        F1 = self.relu(self.conv1(X))
        F2 = self.relu(self.conv3(self.relu(self.conv2(F1)))+F1)

        F3 = self.relu(self.conv4(F2))

        F4 = self.relu(self.conv6(self.relu(self.conv5(F3)))+F3)
        F5 = self.relu(self.conv7(F4))

        F6 = self.relu(self.conv9(self.relu(self.conv8(F5)))+F5)
        F7 = self.relu(self.conv10(F6))

        F8 = self.relu(self.conv12(self.relu(self.conv11(F7)))+F7)
        F9 = self.relu(self.conv13(F8))

        F10 = self.relu(self.conv15(self.relu(self.conv14(F9)))+F9)
        F11 = self.relu(self.conv16(F10))

        F12 = self.relu(self.conv18(self.relu(self.conv17(F11)))+F11)
        F13 = self.relu(self.conv19(F12))

        F14 = self.relu(self.conv21(self.relu(self.conv20(F13)))+F13)

        feat = F14
        Y = self.conv22(F14)
        if collect_feat:
            return Y, feat
        return Y

class TDModel(nn.Module):

    def __init__(self, opts, nc_out, nc_ch):
        super(TDModel, self).__init__()

        self.epoch = 0

        use_bias = True

        self.conv1 = nn.Conv3d(3, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])
        self.conv5 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv7 = nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])
        self.conv8 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv10 = nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])
        self.conv11 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv13 = nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])
        self.conv14 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv15 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv16 = nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])
        self.conv17 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv18 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv19 = nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])
        self.conv20 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv21 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv22 = nn.Conv3d(nc_ch, nc_out, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, X, collect_feat=False):

        F1 = self.relu(self.conv1(X))
        F2 = self.relu(self.conv3(self.relu(self.conv2(F1)))+F1)

        F3 = self.relu(self.conv4(F2))

        F4 = self.relu(self.conv6(self.relu(self.conv5(F3)))+F3)
        F5 = self.relu(self.conv7(F4))

        F6 = self.relu(self.conv9(self.relu(self.conv8(F5)))+F5)
        F7 = self.relu(self.conv10(F6))

        F8 = self.relu(self.conv12(self.relu(self.conv11(F7)))+F7)
        F9 = self.relu(self.conv13(F8))

        F10 = self.relu(self.conv15(self.relu(self.conv14(F9)))+F9)
        F11 = self.relu(self.conv16(F10))

        F12 = self.relu(self.conv18(self.relu(self.conv17(F11)))+F11)
        F13 = self.relu(self.conv19(F12))

        F14 = self.relu(self.conv21(self.relu(self.conv20(F13)))+F13)

        feat = F14
        Y = self.conv22(F14)
        if collect_feat:
            return Y, feat
        return Y

class TDModel_3f_shallow(nn.Module):

    def __init__(self, opts, nc_out, nc_ch):
        super(TDModel_3f_shallow, self).__init__()

        self.epoch = 0

        use_bias = True

        self.conv1 = nn.Conv3d(3, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)#nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[1, 1, 1])
        self.conv5 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv7 = nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])#nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[1, 1, 1])
        self.conv8 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv10 = nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])#nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[1, 1, 1])
        self.conv11 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv13 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)#nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[1, 1, 1])
        # self.conv14 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        # self.conv15 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        # self.conv16 = nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])
        # self.conv17 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        # self.conv18 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        # self.conv19 = nn.Conv3d(nc_ch, nc_ch, kernel_size=[2, 3, 3], stride=1, padding=[0, 1, 1])
        # self.conv20 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        # self.conv21 = nn.Conv3d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv22 = nn.Conv3d(nc_ch, nc_out, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, X, collect_feat=False):

        F1 = self.relu(self.conv1(X))
        F2 = self.relu(self.conv3(self.relu(self.conv2(F1)))+F1)

        F3 = self.relu(self.conv4(F2))

        F4 = self.relu(self.conv6(self.relu(self.conv5(F3)))+F3)
        F5 = self.relu(self.conv7(F4))

        F6 = self.relu(self.conv9(self.relu(self.conv8(F5)))+F5)
        F7 = self.relu(self.conv10(F6))

        F8 = self.relu(self.conv12(self.relu(self.conv11(F7)))+F7)
        F9 = self.relu(self.conv13(F8))

        # F10 = self.relu(self.conv15(self.relu(self.conv14(F9)))+F9)
        # F11 = self.relu(self.conv16(F10))

        # F12 = self.relu(self.conv18(self.relu(self.conv17(F11)))+F11)
        # F13 = self.relu(self.conv19(F12))

        # F14 = self.relu(self.conv21(self.relu(self.conv20(F13)))+F13)

        feat = F9
        Y = self.conv22(F9)
        if collect_feat:
            return Y, feat
        return Y

class TDModel_2d(nn.Module):

    def __init__(self, opts, nc_out, nc_ch):
        super(TDModel_2d, self).__init__()

        self.epoch = 0

        use_bias = True

        self.conv1 = nn.Conv2d(3, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv7 = nn.Conv2d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv10 = nn.Conv2d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv13 = nn.Conv2d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv14 = nn.Conv2d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv15 = nn.Conv2d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv16 = nn.Conv2d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv17 = nn.Conv2d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv18 = nn.Conv2d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv19 = nn.Conv2d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv20 = nn.Conv2d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)
        self.conv21 = nn.Conv2d(nc_ch, nc_ch, kernel_size=3, stride=1, padding=1)

        self.conv22 = nn.Conv2d(nc_ch, nc_out, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, X, collect_feat=False):

        F1 = self.relu(self.conv1(X))
        F2 = self.relu(self.conv3(self.relu(self.conv2(F1)))+F1)

        F3 = self.relu(self.conv4(F2))

        F4 = self.relu(self.conv6(self.relu(self.conv5(F3)))+F3)
        F5 = self.relu(self.conv7(F4))

        F6 = self.relu(self.conv9(self.relu(self.conv8(F5)))+F5)
        F7 = self.relu(self.conv10(F6))

        F8 = self.relu(self.conv12(self.relu(self.conv11(F7)))+F7)
        F9 = self.relu(self.conv13(F8))

        F10 = self.relu(self.conv15(self.relu(self.conv14(F9)))+F9)
        F11 = self.relu(self.conv16(F10))

        F12 = self.relu(self.conv18(self.relu(self.conv17(F11)))+F11)
        F13 = self.relu(self.conv19(F12))

        F14 = self.relu(self.conv21(self.relu(self.conv20(F13)))+F13)
        feat = F14
        Y = self.conv22(F14)
        
        if collect_feat:
            return Y, feat
        return Y

# teacher
# input = torch.ones(1,3,7,128,128)
# model = TDModel(None, 3, 64)
# 127603359744 (127.60G)

# student1
# input = torch.ones(1,3,3,128,128)
# model = TDModel_3f(None, 3, 16)
# 5761449984 (5.76G)

# student2
# input = torch.ones(1,3,3,128,128)
# model = TDModel_3f_shallow(None, 3, 16)
# 2810232832 (2.81G)

# student3
input = torch.ones(1,3,128,128)
model = TDModel_2d(None, 3, 32)
# 3059269632 (3.06G)


flops, params = profile(model, inputs=(input, ))

print(flops)