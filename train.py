import os
import itertools
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
import torch
from models.cycle_gan import Discriminator,Generator,init_params
from models.dataset import ImageDataset
import utils.utils as utils
from PIL import Image
from torch.autograd import Variable

opt=utils.load_config('./configs/CycleGAN_config.yaml')

#------------数据读取-------------
transform = transforms.Compose([
    transforms.Resize(int(opt['Image']['img_height'] * 1.12), Image.BICUBIC),
    transforms.RandomCrop((opt['Image']['img_height'], opt['Image']['img_width'])),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataloader = DataLoader(
    ImageDataset(f"./datasets/{opt['Train']['dataset_name']}", transform=transform,mode='train'),
    batch_size=opt['Train']['batch_size'],
    shuffle=True,
)

val_dataloader = DataLoader(
    ImageDataset(f"./datasets/{opt['Train']['dataset_name']}", transform=transform,mode="test"),
    batch_size=5,
)

device=opt['Train']['device']

# 根据论文原文，损失分为对抗损失、循环损失、恒等映射损失
criterion_GAN = torch.nn.MSELoss().to(device)
criterion_cycle = torch.nn.L1Loss().to(device)
criterion_identity = torch.nn.L1Loss().to(device)

G_AB=Generator(opt['Train']['n_blocks']).to(device)
G_BA=Generator(opt['Train']['n_blocks']).to(device)
D_A=Discriminator().to(device)
D_B=Discriminator().to(device)

for i in [G_AB,G_BA,D_A,D_B]:
    i.apply(init_params)

#--------------优化器---------------
# 一个trick 在一个优化器中设置多个网络参数
optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt['Train']['lr'])
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt['Train']['lr'])
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt['Train']['lr'])

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=utils.LambdaLR(opt["Train"]["n_epochs"], 0, 100).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=utils.LambdaLR(opt["Train"]["n_epochs"], 0, 100).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=utils.LambdaLR(opt["Train"]["n_epochs"], 0, 100).step
)


# Buffers of previously generated samples
fake_A_buffer = utils.ReplayBuffer()
fake_B_buffer = utils.ReplayBuffer()

os.makedirs("images/%s" % opt['Train']['dataset_name'], exist_ok=True)

def sample_images(epoch):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(imgs["A"].to(device))
    fake_B = G_AB(real_A)
    real_B = Variable(imgs["B"].to(device))
    fake_A = G_BA(real_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "images/%s/%s.png" % (opt['Train']['dataset_name'], epoch), normalize=False)


with tqdm(range(opt["Train"]['n_epochs'])) as pbar:
    for epoch in pbar:
        for batch in (dataloader):

            # Set model input
            real_A = Variable(batch["A"].to(device)) #(B,3,256,256)
            real_B = Variable(batch["B"].to(device))

            # 真实标签，后续将计算与其的L2损失
            output_shape = (1, opt["Image"]["img_height"] // 2 ** 4, opt["Image"]["img_width"] // 2 ** 4)
            valid = torch.ones((opt['Train']['batch_size'],*output_shape), requires_grad=False).to(device) #(B,1,16,16) 和辨别器的输出一致
            fake = torch.ones((opt['Train']['batch_size'],*output_shape), requires_grad=False).to(device)  #(B,1,16,16)

            # ------------------
            #  Train Generators
            # ------------------

            G_AB.train()
            G_BA.train()

            optimizer_G.zero_grad()

            # 恒等映射的损失
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # 对抗损失
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # 循环损失
            recov_A = G_BA(fake_B) # A-B-A
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A) # B-A-B
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # 总损失
            loss_G = loss_GAN + opt["Train"]["lambda_cyc"] * loss_cycle + opt["Train"]["lambda_identity"]* loss_identity

            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            optimizer_D_A.zero_grad()

            # 辨别器A要尽可能对真A辨别真
            loss_real = criterion_GAN(D_A(real_A), valid)
            # 辨别器A要对过去的假A辨别为假
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            # 总损失
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------

            optimizer_D_B.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

             # Update learning rates
            lr_scheduler_G.step()
            lr_scheduler_D_A.step()
            lr_scheduler_D_B.step()
       
            # Print log
            pbar.set_postfix(
                {
                    "Epoch:":epoch,
                    "D loss":f'{loss_D.item():.3f}',
                    "G loss":f'{loss_G.item():.3f}',
                    "loss GAN":f'{loss_GAN.item():.3f}',
                    "loss cyc":f"{loss_cycle.item():.3f}",
                    "loss identity":f"{loss_identity.item():.3f}",
                }
            )

            # If at sample interval save image
            if epoch % opt["Train"]["sample_interval"] == 0:
                sample_images(epoch)

        if epoch % opt["Train"]["checkpoint_interval"] == 0:
            # Save model checkpoints
            torch.save(G_AB.state_dict(), "pretrained_models/%s/G_AB_%d.pth" % (opt['Train']['dataset_name'], epoch))
            torch.save(G_BA.state_dict(), "pretrained_models/%s/G_BA_%d.pth" % (opt['Train']['dataset_name'], epoch))
            torch.save(D_A.state_dict(), "pretrained_models/%s/D_A_%d.pth" % (opt['Train']['dataset_name'], epoch))
            torch.save(D_B.state_dict(), "pretrained_models/%s/D_B_%d.pth" % (opt['Train']['dataset_name'], epoch))