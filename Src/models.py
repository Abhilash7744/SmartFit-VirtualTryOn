import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, person_img, garment_img):
        input_tensor = torch.cat([person_img, garment_img], dim=1)
        return self.model(input_tensor)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 64 * 64, 1),
            nn.Sigmoid()
        )

    def forward(self, image):
        return self.model(image)
