import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()

        # Encoder blocks with downsampling
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        )

        self.encoder_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7
        )

        self.encoder_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )

        self.encoder_block4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )

        self.flatten = nn.Flatten()
        
        self.mean_linear = nn.Linear(in_features=64 * 7 * 7, out_features=latent_dim)
        self.log_var_linear = nn.Linear(in_features=64 * 7 * 7, out_features=latent_dim)

        # Decoder blocks with upsampling
        self.decoder_linear = nn.Linear(in_features=latent_dim, out_features=64 * 7 * 7)
        
        self.decoder_block1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder_block2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder_block3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True),  # 7x7 -> 14x14
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder_block4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1, bias=True),  # 14x14 -> 28x28
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder_block1(x)
        x = self.encoder_block2(x)
        x = self.encoder_block3(x)
        x = self.encoder_block4(x)
        x = self.flatten(x)
        return self.mean_linear(x), self.log_var_linear(x)
    
    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        z = self.decoder_linear(z)
        z = z.view(-1, 64, 7, 7)  # This is now correct
        z = self.decoder_block1(z)
        z = self.decoder_block2(z)
        z = self.decoder_block3(z)
        z = self.decoder_block4(z)
        return z
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        print(mu.shape, log_var.shape)
        z = self.sample(mu, log_var)
        print(z.shape)
        return self.decode(z), mu, log_var
    
def mse_loss(x, x_hat):
    return torch.nn.functional.mse_loss(x, x_hat)

def kl_divergence(mu, log_var):
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

def total_loss(x, x_hat, mu, log_var):
    batch_size = x.size(0)
    return mse_loss(x, x_hat) + kl_divergence(mu, log_var) / batch_size



dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)

model = VAE(latent_dim=20)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

for epoch in range(10):
    model.train()
    all_loss = 0
    print(f"Epoch {epoch+1}/10")
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (data, _) in loop:
        data = data.to(device)
        x_hat, mu, log_var = model(data)
        loss = total_loss(data, x_hat, mu, log_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = all_loss / len(train_loader)
    print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}\n")




with torch.no_grad():
    model.eval()
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    data_iter = iter(train_loader)
    x, _ = next(data_iter)
    x = x[:5].to(device) 
    x_hat, mu, log_var = model(x)
    
    for i in range(5):
        axes[i].imshow(x_hat[i].view(28, 28).cpu().detach().numpy(), cmap='gray')
        axes[i].set_title(f'Reconstructed {i+1}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()









        
        
        
        
        
       