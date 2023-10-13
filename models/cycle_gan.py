import torch
from torch import nn


def init_params(m:nn.Module):
    class_name=m.__class__.__name__ # 所属类名 例如Conv2d
    if class_name.find("Conv") != -1:
        # 寻找子串 是否为卷积层
        torch.nn.init.normal_(m.weight,0,0.02)
        if hasattr(m,"bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias,0.0) # 将偏置初始化为0

    elif class_name.find("BatchNorm2d") !=-1:
        torch.nn.init.normal_(m.weight,1,0.02)
        torch.nn.init.constant_(m.bias,0.0)

class Generator(nn.Module):
    """
    3个卷积层降采样+若干个残差块+3个卷积层反卷积升采样 输入输出维度一致
    """
    def __init__(self,n_blocks=9) -> None:
        super().__init__()
        # 卷积层：获取特征图
        def conv_block(in_channels,out_channels,is_transposed=False):
            
            layers=[
                nn.Conv2d(in_channels,out_channels,3,2,1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(True),
            ]
            if is_transposed:
                layers[0]=nn.ConvTranspose2d(in_channels,out_channels,3,2,padding=1,output_padding=1)
            return layers
        
        self.conv_net=nn.Sequential(
            nn.ReflectionPad2d(3), # 因为-7+1+6保证维度不变
            nn.Conv2d(3,64,7,1,0),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            *conv_block(64,128),  # 步长为2，核大小为3 维度减半
            *conv_block(128,256), # 维度减半
        )

        self.resnet=nn.Sequential(*[ResidualBlock(256) for _ in range(n_blocks)])

        self.conv_transposed_net=nn.Sequential(
            *conv_block(256,128,True),
            *conv_block(128,64,True),
            nn.ReflectionPad2d(3), # 因为-7+1+6保证维度不变
            nn.Conv2d(64,3,7,1,0),
            nn.Tanh(),
        )
    
    def forward(self,x):
        x=self.conv_net(x) #->(256,64,64)
        x=self.resnet(x) #->(256,64,64)
        x=self.conv_transposed_net(x) #->(3,256,256)
        return x



class ResidualBlock(nn.Module):
    """
    残差块，负责提取特征
    由于cyclegan的中游部分是残差块构成，通道数没有改变，输入输出维度一致
    因此参数只有一个通道数
    """
    def __init__(self,channels) -> None:
        super().__init__()
        
        self.conv_block=nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels,channels,kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), # 镜像反射填充！
            nn.Conv2d(channels,channels,3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self,x):
        output=x+self.conv_block(x)
        return output

class Discriminator(nn.Module):
    """
    PatchGan的结构，全部为4x4的卷积 架构为3-64-128-256-512-1
    InstanceNorm2d 每个特征图的的每个通道的所有像素点归一化
    注意z轴是H*W 即展平结果
    """
    def __init__(self):
        super().__init__()

        def discriminator_block(in_channels,out_channels):
            layers=[
                nn.Conv2d(in_channels,out_channels,kernel_size=4,stride=2,padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2,inplace=True),
                ] # 长宽减半
            return layers
        
        self.net=nn.Sequential(
            *discriminator_block(3,64),
            *discriminator_block(64,128),
            *discriminator_block(128,256),
            *discriminator_block(256,512),
            nn.ZeroPad2d((1,0,1,0)), # 只填充左和上 因为(16-4+1+2)会变为15 只有在两边填充才能避免 而且必须是相邻两侧 不然输出是(1,15,17)这样
            nn.Conv2d(512,1,4,padding=1),
        )


    def forward(self,img):
        return self.net(img)
    
if __name__ == '__main__':
    gene=Generator(9)
    test=torch.randn(2,3,256,256)
    print(gene(test).shape)
