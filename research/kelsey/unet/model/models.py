"""
Aggregating all the UNet model code into one script
Code refactored from: https://github.com/milesial/Pytorch-UNet/tree/master semantic
segmentation implementation
"""

from model.unet_modules import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = (DoubleConv(n_channels, 32))     # input image size selecting 32 as smallest
        self.down1 = (Down(32, 64))                 # doubling feature channels
        self.down2 = (Down(64, 128))                # doubling feature channels
        self.down3 = (Down(128, 256))               # doubling feature channels
        self.down4 = (Down(256, 512 // 2))
        self.up1 = (Up(512, 256 // 2))              # upsampling, halving number of features
        self.up2 = (Up(256, 128 // 2))              # upsampling, halving number of features
        self.up3 = (Up(128, 64 // 2))               # upsampling, halving number of features
        self.up4 = (Up(64, 32))                     # supsampling, halving the number of features
        self.outc = (OutConv(32, n_classes))        # final output matches input size with number of classes specified (1 for regressiom)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        output = self.outc(x9)
        return output


    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class MCDropoutProbabilisticUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(MCDropoutProbabilisticUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = (DoubleConv(n_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        self.down4 = (Down(256, 512 // 2))
        self.up1 = (Up(512, 256 // 2))
        self.up2 = (Up(256, 128 // 2))
        self.up3 = (Up(128, 64 // 2))
        self.up4 = (Up(64, 32))
        self.outc = (OutConv(32, n_classes))
        self.log_var = (OutConv(32, n_classes))
        self.drop = torch.nn.Dropout2d(p=0.1)


    def forward(self, x):
        x1 = self.drop(self.inc(x))
        x2 = self.drop(self.down1(x1))
        x3 = self.drop(self.down2(x2))
        x4 = self.drop(self.down3(x3))
        x5 = self.drop(self.down4(x4))
        x6 = self.drop(self.up1(x5, x4))
        x7 = self.drop(self.up2(x6, x3))
        x8 = self.drop(self.up3(x7, x2))
        x9 = self.drop(self.up4(x8, x1))
        output = self.drop(self.outc(x9))        # Applying dropout to output for mcdropout
        log_var = self.drop(self.log_var(x9))    # Applying dropout to output for mcdropout
        return output, log_var


    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        self.log_var = torch.utils.checkpoint(self.log_var)


'''
class ConcreteDropoutProbabilisticUNet(nn.Module):
    def __init__(self, n_channels, n_classes,
                 dropout_regularizer=1e-5):
        super(ConcreteDropoutProbabilisticUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.dropout_regularizer = dropout_regularizer

        self.inc = (DropoutDoubleConv(n_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        self.down4 = (Down(256, 512 // 2))
        self.up1 = (Up(512, 256 // 2))
        self.up2 = (Up(256, 128 // 2))
        self.up3 = (Up(128, 64 // 2))
        self.up4 = (Up(64, 32))
        self.outc = (OutConv(32, n_classes))


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.outc(x)
        return output


    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
'''