import glob # 用于文件和目录管理：
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    """
    读取两个文件夹下的所有图片,每次获得两张图
    由于cyclegan不要求数据对齐，因此两种图可以不是配对的
    1. 定位两个文件夹，读取各自所有的图片名并拼接路径
    2. 按顺序遍历其中一个数据集，每次随机抽选另一个数据集的图片

    数据集的文件格式需要是 root/modeA 和root/modeB 
    
    返回
    A (tensor) (C,H,W)      
    B (tensor)

    """
    def __init__(self,root:str,transform=None,mode='train') -> None:
        super().__init__()
        # glob库可以通过*进行通配符，在内部os.path.join可以作用到列表的每个元素
        # 返回：A或B域的所有图片路径
        # 需要注意的是，当前工作目录必须在Cyclegan中，即和当前文件同级
        self.files_A=glob.glob(os.path.join(root,f'{mode}A/*.*'))
        self.files_B=glob.glob(os.path.join(root,f'{mode}B/*.*'))
        self.transform=transform
        if not self.transform: # 没有转换器
            self.transform=transforms.Compose([transforms.PILToTensor()])

    def __getitem__(self,index):
        # 保证索引一定在文件范围中
        img_A=Image.open(self.files_A[index % len(self.files_A)]).convert("RGB")
        # 随机化目标域的B索引，避免总是出现固定的配对输入模型
        img_B=Image.open(self.files_B[random.randint(0,len(self.files_B)-1)]).convert('RGB') # randint包括末尾
     
        img_A=self.transform(img_A)
        img_B=self.transform(img_B)
        return {"A":img_A,"B":img_B}
    
    def __len__(self):
        return max(len(self.files_A),len(self.files_B))
    

if __name__ == '__main__':
    dataset=ImageDataset('datasets/horse2zebra')
    output=dataset.__getitem__(0)
    print(output['A'].shape)


