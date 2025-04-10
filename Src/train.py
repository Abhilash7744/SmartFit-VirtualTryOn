import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from models import Generator, Discriminator
from dataset_preparation import preprocess_images
from PIL import Image
import os

EPOCHS = 20
LEARNING_RATE = 0.0002
BATCH_SIZE = 4

# Load dataset
dataset_dir = "../dataset/processed"
images = [os.path.join(dataset_dir, img) for img in os.listdir(dataset_dir)]
transform = transforms.Compose([transforms.ToTensor()])

class CustomDataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(images)

    def __getitem__(self, idx):
        img = Image.open(images[idx]).convert("RGB")
        return transform(img)

dataloader = DataLoader(CustomDataset(), batch_size=BATCH_SIZE, shuffle=True)

# Initialize models
generator = Generator()
discriminator = Discriminator()
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    for real_images in dataloader:
        # Train Discriminator
        optimizer_D.zero_grad()
        real_labels = torch.ones(real_images.size(0), 1)
        fake_images = generator(real_images, real_images)
        fake_labels = torch.zeros(real_images.size(0), 1)
        real_loss = criterion(discriminator(real_images), real_labels)
        fake_loss = criterion(discriminator(fake_images.detach()), fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        g_loss = criterion(discriminator(fake_images), real_labels)
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch {epoch+1}/{EPOCHS}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

# Save models
torch.save(generator.state_dict(), "../models/generator.pth")
torch.save(discriminator.state_dict(), "../models/discriminator.pth")
print("Training completed! Models saved.")
