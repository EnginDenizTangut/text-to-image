import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class FlowerDataset(Dataset):
    def __init__(self, image_dir, labels, transform=None):
        self.image_dir = image_dir
        self.labels = labels
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx % len(self.labels)]  

        if self.transform:
            image = self.transform(image)

        return image, label

def sentence_to_glove_vector(sentences, embedding_dim=300):

    batch_size = len(sentences)
    return torch.randn(batch_size, embedding_dim)

class Generator(nn.Module):
    def __init__(self, z_dim, embedding_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim + embedding_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 3 * 128 * 128)  

    def forward(self, z, embedding):

        x = torch.cat([z, embedding], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = x.view(x.size(0), 3, 128, 128)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(3 * 128 * 128, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, img):
        x = img.view(img.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

def train(generator, discriminator, dataloader, num_epochs=20, z_dim=100, lr=0.0002, beta1=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        for i, (real_imgs, labels) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            optimizer_D.zero_grad()

            real_validity = discriminator(real_imgs)
            d_loss_real = criterion(real_validity, real_labels)

            z = torch.randn(batch_size, z_dim, device=device)
            embeddings = sentence_to_glove_vector(labels).to(device)
            fake_imgs = generator(z, embeddings)
            fake_validity = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(fake_validity, fake_labels)

            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            z = torch.randn(batch_size, z_dim, device=device)
            embeddings = sentence_to_glove_vector(labels).to(device)
            fake_imgs = generator(z, embeddings)
            fake_validity = discriminator(fake_imgs)
            g_loss = criterion(fake_validity, real_labels)

            g_loss.backward()
            optimizer_G.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], "
                      f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

image_dir = '/content/flowers102/jpg'  
labels = ['rose', 'tulip', 'sunflower'] * 1000  

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = FlowerDataset(image_dir=image_dir, labels=labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

generator = Generator(z_dim=100, embedding_dim=300)
discriminator = Discriminator()

train(generator, discriminator, dataloader, num_epochs=100)

torch.save(generator.state_dict(), "generator_50.pth")
torch.save(discriminator.state_dict(), "discriminator_50.pth")
print("Modeller kaydedildi: generator_50.pth ve discriminator_50.pth")

import torch
from torchvision.utils import save_image

generator = Generator(z_dim=100, embedding_dim=300).to('cuda' if torch.cuda.is_available() else 'cpu')
generator.load_state_dict(torch.load("generator_50.pth", map_location=torch.device('cpu')))  
generator.eval()  

def generate_flower(label):
    z = torch.randn(1, 100).to('cuda' if torch.cuda.is_available() else 'cpu')  
    embedding = sentence_to_glove_vector([label]).to('cuda' if torch.cuda.is_available() else 'cpu')  
    with torch.no_grad():  
        fake_image = generator(z, embedding)

    save_image(fake_image * 0.5 + 0.5, f"{label.replace(' ', '_')}.png")  
    print(f"{label}_50.png kaydedildi!")

generate_flower("sunflower")
generate_flower("mavi orkide")