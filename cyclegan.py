import torch
import torch.nn
import torchvision
import PIL.Image
import pathlib
import matplotlib.pyplot
from torch.utils.data import DataLoader
from torchvision import datasets

device = torch.device("cuda")
batch_size = 3
reconstruct_weight = 10
id_weight = 1

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256,256)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

def res(inputc, outputc):
    return torch.nn.Sequential(
        torch.nn.Conv2d(inputc, outputc, kernel_size=3, stride=1, padding=1),
        torch.nn.InstanceNorm2d(num_features = outputc),
        torch.nn.LeakyReLU(negative_slope=0.01),
        torch.nn.Conv2d(inputc, outputc, kernel_size=3, stride=1, padding=1),
        torch.nn.InstanceNorm2d(num_features = outputc)
    )

def downsample(inputc, outputc):
    return torch.nn.Sequential(
        torch.nn.Conv2d(inputc, inputc, kernel_size=3, stride=1, padding=1),
        torch.nn.Conv2d(inputc, outputc, kernel_size=2, stride=2),
        torch.nn.InstanceNorm2d(num_features = outputc),
        torch.nn.LeakyReLU(negative_slope=0.01)
    )

def upsample(inputc, outputc):
    return torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inputc, outputc, kernel_size=4, stride=2, padding=1),
        torch.nn.Conv2d(outputc, outputc, kernel_size=3, stride=1, padding=1),
        torch.nn.InstanceNorm2d(num_features = outputc),
        torch.nn.LeakyReLU(negative_slope=0.01)
    )

#using UNet(input : batch_size*3*128*128)
class G(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.res1 = res(256,256)
        self.res2 = res(256,256)
        self.res3 = res(256,256)
        self.res4 = res(256,256)
        self.res5 = res(256,256)
        self.res6 = res(256,256)
        self.res7 = res(256,256)
        self.res8 = res(256,256)
        self.res9 = res(256,256)
        self.up1 = torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.up2 = torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.up3 = torch.nn.ConvTranspose2d(64, 3, kernel_size=7, stride=1, padding=3)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = self.res1(x)+x
        x = self.res2(x)+x
        x = self.res3(x)+x
        x = self.res4(x)+x
        x = self.res5(x)+x
        x = self.res6(x)+x
        x = self.res7(x)+x
        x = self.res8(x)+x
        x = self.res9(x)+x
        x = torch.nn.functional.relu(self.up1(x))
        x = torch.nn.functional.relu(self.up2(x))
        x = self.up3(x) 
        return torch.nn.functional.tanh(x)
        
#input : batch_size*3*128*128
class D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = downsample(3, 32)
        self.d2 = downsample(32, 64)
        self.d3 = downsample(64, 128)
        self.d4 = downsample(128, 256)
        self.output = torch.nn.Conv2d(256, 1, kernel_size=3, stride=1, padding = 1)
        
    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)
        return torch.nn.functional.sigmoid(self.output(x4))

class CycleGan(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.G_AB = G()
        self.G_BA = G()
        self.D_A = D()
        self.D_B = D()
    
    def freeze_D(self):
        for p in self.D_A.parameters():
            p.requires_grad = False
        for p in self.D_B.parameters():
            p.requires_grad = False
            
    def unfreeze_D(self):
        for p in self.D_A.parameters():
            p.requires_grad = True
        for p in self.D_B.parameters():
            p.requires_grad = True
    
    def forward(self, imga, imgb):
        fake_A = self.G_BA(imgb)
        fake_B = self.G_AB(imga)
        
        valid_A = self.D_A(fake_A)
        valid_B = self.D_B(fake_B)
        
        reconstruct_A = self.G_BA(fake_B)
        reconstruct_B = self.G_AB(fake_A)
        
        id_A = self.G_BA(imga)
        id_B = self.G_AB(imgb)
        
        return (fake_A, fake_B, valid_A, valid_B, reconstruct_A, reconstruct_B, id_A, id_B)
    
        
class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        X = pathlib.Path("./data/style_x/")
        Y = pathlib.Path("./data/style_y/")
        self.X_data = [str(i) for i in X.iterdir()]
        self.Y_data = [str(i) for i in Y.iterdir()]
        
    def __len__(self):
        return min(len(self.X_data), len(self.Y_data))
    
    def __getitem__(self, idx):
        return (transform(PIL.Image.open(self.X_data[idx]).convert('RGB')).to(device), transform(PIL.Image.open(self.Y_data[idx]).convert('RGB')).to(device))
#test======================================================================================================================
def test(model, imga, imgb, save_path="output.png"):
    model.eval()
    output1, output2 = model(imga, imgb)[0:2]
    output1 = (output1+1)/2
    output2 = (output2+1)/2
    fig, ax = matplotlib.pyplot.subplots(2, 2)
    ax[0,0].imshow(output1.detach().squeeze().permute(1, 2, 0).cpu())
    ax[0,1].imshow(output2.detach().squeeze().permute(1, 2, 0).cpu())     
    ax[1,0].imshow((imga.squeeze().permute(1,2,0).cpu()+1)/2)
    ax[1,1].imshow((imgb.squeeze().permute(1,2,0).cpu()+1)/2)
    
    matplotlib.pyplot.savefig(save_path, dpi=300, bbox_inches='tight')
    matplotlib.pyplot.close() 
#G train ======================================================================================================================
# this loss function is for calculating out the total loss of generator
def loss_G(imga, imgb, valid_A, valid_B, reconstruct_A, reconstruct_B, id_A, id_B):
    return 10*torch.nn.functional.mse_loss(valid_A, torch.ones_like(valid_A))+ \
    10*torch.nn.functional.mse_loss(valid_B, torch.ones_like(valid_B)) + \
    reconstruct_weight*torch.nn.functional.l1_loss(imga, reconstruct_A) + \
    reconstruct_weight*torch.nn.functional.l1_loss(imgb, reconstruct_B) + \
    id_weight*torch.nn.functional.l1_loss(imga, id_A)  + \
    id_weight*torch.nn.functional.l1_loss(imgb, id_B)

#this train funciton if for training the generators
def train_one_epoch_G(model, optimizer, imga, imgb):
    fake_A, fake_B, valid_A, valid_B, reconstruct_A, reconstruct_B, id_A, id_B = model(imga, imgb)
    LOSS = loss_G(imga, imgb, valid_A, valid_B, reconstruct_A, reconstruct_B, id_A, id_B)
    optimizer.zero_grad()
    LOSS.backward()
    optimizer.step()
    return LOSS.item()

def train_G(model, optimizer, loader, epoches):
    model.freeze_D()
    tot = 0
    num = 0
    for i in range(epoches):
        for id,(x,y) in enumerate(loader):
            tot += train_one_epoch_G(model, optimizer, x, y)
            num += 1
            
    model.unfreeze_D()
    print("G is training, loss: ", (tot/num))
    
#D train ======================================================================================================================
# this function is specific for training Disctiminator
def train_D(model, optimizer_DA, optimizer_DB, loader, epoches):
    model.unfreeze_D()
    for epoch in range(epoches):
        for idx, (x, y) in enumerate(loader):
            fake_A = model.G_BA(y).detach()
            fake_B = model.G_AB(x).detach()

            # ===== D_A =====
            pred_real_A = model.D_A(x)
            pred_fake_A = model.D_A(fake_A)
            loss_real_A = torch.nn.functional.mse_loss(
                pred_real_A,
                torch.ones_like(pred_real_A)
            )
            loss_fake_A = torch.nn.functional.mse_loss(
                pred_fake_A,
                torch.zeros_like(pred_fake_A)
            )
            loss_DA = (loss_real_A + loss_fake_A) / 2
            optimizer_DA.zero_grad()
            loss_DA.backward()
            optimizer_DA.step()

            # ===== D_B =====
            pred_real_B = model.D_B(y)
            pred_fake_B = model.D_B(fake_B)
            loss_real_B = torch.nn.functional.mse_loss(
                pred_real_B,
                torch.ones_like(pred_real_B)
            )
            loss_fake_B = torch.nn.functional.mse_loss(
                pred_fake_B,
                torch.zeros_like(pred_fake_B)
            )
            loss_DB = (loss_real_B + loss_fake_B) / 2
            optimizer_DB.zero_grad()
            loss_DB.backward()
            optimizer_DB.step()
        print(
            f"D_A loss:{loss_DA.item():.4f}, "
            f"D_B loss:{loss_DB.item():.4f}"
        )

    
#model = CycleGan().to(device)
model = torch.load("model/model.pth", weights_only = False)
optimizer_G = torch.optim.Adam(list(model.G_AB.parameters()) + list(model.G_BA.parameters()), lr=0.0001)
optimizer_DA = torch.optim.Adam(model.D_A.parameters(), lr = 0.00001)
optimizer_DB = torch.optim.Adam(model.D_B.parameters(), lr = 0.00001)
my_dataset = MyDataset()
loader = torch.utils.data.DataLoader(my_dataset, shuffle = True, batch_size = batch_size)

for i in range(100):
    train_D(model, optimizer_DA, optimizer_DB, loader, 1)
    train_G(model, optimizer_G, loader, 1)
    torch.save(model, "model/model.pth")
    
    test(model,transform(PIL.Image.open("data_hor/style_x/n02381460_117.jpg").convert('RGB')).unsqueeze(0).to(device),
    transform(PIL.Image.open("data_hor/style_y/n02391049_45.jpg").convert('RGB')).unsqueeze(0).to(device))
    
