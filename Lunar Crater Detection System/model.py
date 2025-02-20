#import the dependencies
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from PIL import Image

#custokom dataset class
from torchvision import transforms
import os

class LunarCraterDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None, target_format="png"):
        self.image_paths = sorted([
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir) if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ])
        self.mask_paths = sorted([
            os.path.join(mask_dir, fname)
            for fname in os.listdir(mask_dir) if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ])
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.target_format = target_format

    def __len__(self):
        return len(self.image_paths)

    #if necessary. if other format
    def _convert_to_target_format(self, img):
        if self.target_format and not img.format.lower() == self.target_format:
            img = img.convert("RGB")  # RGB
        return img

    def __getitem__(self, idx):
        # Load
        image = Image.open(self.image_paths[idx])
        mask = Image.open(self.mask_paths[idx])

        # Convert to target format(if neces)
        image = self._convert_to_target_format(image)
        mask = mask.convert("L")#grayscale

        #transformationss
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask



# Data transformations
#image
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# masks
mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

#load data,create DataLoade
from torch.utils.data import DataLoader


train_image_dir = "train_image_dir"
train_mask_dir = "train_mask_dir"
val_image_dir = "test_image_dir"
val_mask_dir = "test_image_dir"

#  Dataset
train_dataset = LunarCraterDataset(train_image_dir, train_mask_dir, image_transform=image_transform, mask_transform=mask_transform)
val_dataset = LunarCraterDataset(val_image_dir, val_mask_dir, image_transform=image_transform, mask_transform=mask_transform)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

#Unet architecture with ResNet backbone
class ResNetUNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNetUNet, self).__init__()

        # Pretrained ResNet18 (imaggenet dataset)
        base_model = models.resnet18(pretrained=True)
        self.base_layers = list(base_model.children())

        # Encoder
        self.enc1 = nn.Sequential(*self.base_layers[:3])
        self.enc2 = nn.Sequential(*self.base_layers[3:5])
        self.enc3 = self.base_layers[5]
        self.enc4 = self.base_layers[6]
        self.enc5 = self.base_layers[7]

        # Decoder
        self.up5 = self._upsample(512, 256)
        self.up4 = self._upsample(256, 128)
        self.up3 = self._upsample(128, 64)
        self.up2 = self._upsample(64, 64)
        self.up1 = self._upsample(64, 32)

        # Final
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def _upsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # Encoder forward pass
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        # Decoder forward pass
        dec5 = self.up5(enc5)
        dec4 = self.up4(dec5 + enc4)
        dec3 = self.up3(dec4 + enc3)
        dec2 = self.up2(dec3 + enc2)
        dec1 = self.up1(dec2 + enc1)

        return self.final(dec1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetUNet(num_classes=2).to(device)

import os
from tqdm import tqdm

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Save checkpoint
def save_checkpoint(model, optimizer, epoch, step, checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}_step{step}.pth")
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

# Load checkpoint
def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    start_step = checkpoint['step']
    print(f"Resumed from checkpoint: {checkpoint_path} (Epoch: {start_epoch}, Step: {start_step})")
    return start_epoch, start_step

# Train
def train_one_epoch_with_checkpoint(model, train_loader, criterion, optimizer, device, epoch, checkpoint_steps=500):
    model.train()
    epoch_loss = 0
    step = 0

    with tqdm(train_loader, desc=f"Training Epoch {epoch+1}", unit="batch") as pbar:
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1).long())

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            step += 1

            pbar.set_postfix({"loss": loss.item(), "step": step})

            # Save checkpoint every 500 steps
            if step % checkpoint_steps == 0:
                save_checkpoint(model, optimizer, epoch, step)

    return epoch_loss / len(train_loader)

# Validation
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0

    with tqdm(val_loader, desc="Validation", unit="batch") as pbar:
        with torch.no_grad():
            for images, masks in pbar:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks.squeeze(1).long())
                val_loss += loss.item()

                pbar.set_postfix({"val_loss": loss.item()})

    return val_loss / len(val_loader)

num_epochs = 15
best_loss = float('inf')
start_epoch = 0
start_step = 0

# Checkpoint path to resume from
checkpoint_path = ""  # Replace with the actual path if resuming
if checkpoint_path:
    start_epoch, start_step = load_checkpoint(model, optimizer, checkpoint_path)

step = start_step
for epoch in range(start_epoch, num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    # Training
    train_loss = train_one_epoch_with_checkpoint(
        model, train_loader, criterion, optimizer, device, epoch, checkpoint_steps=500
    )

    # Validation
    val_loss = validate(model, val_loader, criterion, device)
    print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Save checkpointp
    save_checkpoint(model, optimizer, epoch, step)

#load
def load_model_for_inference(model, checkpoint_path, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from checkpoint '{checkpoint_path}' for inference.")

    return model

#load from checkpoint
checkpoint_path = '/content/checkpoint_epoch2_step500.pth'#replace with actual path
model = load_model_for_inference(model, checkpoint_path, device)

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# Transform
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load image
image_path = '/content/drive/MyDrive/images/100_0_png.rf.d8cf68fffa427fdd46f70a8dfbdde753.jpg'
image = Image.open(image_path).convert('RGB')  # RGB


# Display the loaded image
plt.imshow(image)
plt.title("Loaded Image")
plt.axis('off')  # Hide axes for better display
plt.show()


# Transform
image_tensor = image_transform(image).unsqueeze(0).to(device)

import torch
import matplotlib.pyplot as plt
from torchvision import transforms

#mage_tensor = transforms.ToTensor()(image).unsqueeze(0)  # Convert image to tensor and add batch dimension
model.to(device)
with torch.no_grad():
    outputs = model(image_tensor)

predicted_mask = torch.argmax(outputs, dim=1)
predicted_mask = predicted_mask.squeeze(0).cpu().numpy()

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(image)  # Assuming `image` is a PIL image
ax[0].set_title("Input Image")
ax[0].axis('off')

ax[1].imshow(predicted_mask, cmap='gray')
ax[1].set_title("Predicted Mask")
ax[1].axis('off')

plt.show()

from PIL import Image, ImageDraw
import numpy as np
from scipy import ndimage
from IPython.display import display

def process_and_display(binary_mask_array, original_image):
    if binary_mask_array.dtype != np.bool_:
        binary_mask_array = binary_mask_array > 0

    #binary mask array toPIL Image
    mask = Image.fromarray((binary_mask_array * 255).astype(np.uint8))
    original_image = original_image.convert("RGBA")

    mask = mask.resize(original_image.size)

    mask_array = np.array(mask) > 0

    # Create a transparent image for bounding boxes
    transparent_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(transparent_image)

    # Find bounding boxes in the binary mask
    labeled, num_features = ndimage.label(mask_array)
    objects = ndimage.find_objects(labeled)

    # Draw bounding boxes around white regions
    for obj_slice in objects:
        x_min, x_max = obj_slice[1].start, obj_slice[1].stop
        y_min, y_max = obj_slice[0].start, obj_slice[0].stop
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=(255, 0, 0, 255), width=2)  # Red boxes

    # Superimpose
    result = Image.alpha_composite(original_image, transparent_image)


    display(result)
process_and_display(predicted_mask, image)