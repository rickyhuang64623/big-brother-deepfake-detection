import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

class TemplateGenerator(nn.Module):
    def __init__(self, latent_dim=128, output_size=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_size = output_size
        
        # Initial dense layer
        self.fc = nn.Linear(latent_dim, 8 * 8 * 128)
        
        # Simplified upsampling layers
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=False),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2, inplace=False),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(32, 16, 4, 2, 1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.2, inplace=False),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(16, 3, 4, 2, 1),
                nn.Tanh()
            )
        ])
        
    def forward(self, z=None, batch_size=1):
        if z is None:
            z = torch.randn(batch_size, self.latent_dim).to(next(self.parameters()).device)
            
        x = self.fc(z)
        x = x.view(-1, 128, 8, 8)
        
        for layer in self.decoder:
            x = layer(x)
            
        return F.interpolate(x, size=(self.output_size, self.output_size))

class TemplateLoss(nn.Module):
    def __init__(self, lambda_reg=0.3, avg_face_path=None):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.avg_face = None
        if avg_face_path is not None and os.path.exists(avg_face_path):
            avg_face = torch.load(avg_face_path)
            self.avg_face = avg_face
    def forward(self, template, target_template=None):
        # Regularization to keep template close to average face if available
        if self.avg_face is not None:
            avg_face = self.avg_face.to(template.device)
            reg_loss = self.l1_loss(template, avg_face.expand_as(template))
        else:
            reg_loss = self.l1_loss(template, torch.zeros_like(template))
        # Encourage bounded values for stability
        smoothness_loss = self.mse_loss(template, torch.tanh(template))
        if target_template is not None:
            sim_loss = self.mse_loss(template, target_template)
            total_loss = self.lambda_reg * reg_loss + smoothness_loss + sim_loss
        else:
            total_loss = self.lambda_reg * reg_loss + smoothness_loss
            sim_loss = torch.tensor(0.0).to(template.device)
        return total_loss, reg_loss, sim_loss

class LearnableTemplate(nn.Module):
    def __init__(self, size=128, channels=3):
        super().__init__()
        self.size = size
        self.channels = channels
        avg_face_path = os.path.join(os.path.dirname(__file__), '..', 'average_face.pt')
        if os.path.exists(avg_face_path):
            avg_face = torch.load(avg_face_path)
            if avg_face.shape == (1, channels, size, size):
                self.template = nn.Parameter(avg_face.clone())
                print(f"[INFO] Initialized template from average_face.pt")
            else:
                print(f"[WARNING] average_face.pt shape mismatch, using random noise.")
                self.template = nn.Parameter(torch.randn(1, channels, size, size) * 0.1)
        else:
            print(f"[WARNING] average_face.pt not found, initializing template with random noise.")
            self.template = nn.Parameter(torch.randn(1, channels, size, size) * 0.1)
        
    def forward(self, batch_size=1):
        return torch.tanh(self.template.expand(batch_size, -1, -1, -1)) 