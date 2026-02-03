import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torchvision import transforms
from unet import MiniUNet  
from dataset import PetDataset
from utils import combined_loss
import os
import matplotlib.pyplot as plt
import time

# --- CONFIGURAZIONE ---
BATCH_SIZE = 8 
LEARNING_RATE = 1e-3 
NUM_EPOCHS = 25     
DATA_PATH = './data'
MODEL_SAVE_PATH = './models/model_robust.pth'

# --- AUGMENTATION ---
robust_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    
    # 1. Rotazione e Flip (Per i gatti storti/sdraiati)
    transforms.RandomRotation(degrees=20),
    transforms.RandomHorizontalFlip(p=0.5),
    
    # 2. Contrasto e Colore (Per gli animali scuri su sfondo scuro)
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    
    # 3. Sfocatura (Per le texture complesse come il gatto tigrato)
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
    
    transforms.ToTensor(),
    
    # 4. Rumore (Regolarizzazione generale)
    transforms.Lambda(lambda x: x + 0.02 * torch.randn_like(x)),
    
    # Normalizzazione Standard
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.PILToTensor()
])

def train_robust():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🛡️  Avvio Training Robusto Scientifico (Scheduler + Augmentation) su {device}...")

    # 1. Dataset
    full_dataset = PetDataset(root_dir=DATA_PATH, split='trainval')
    full_dataset.img_transform = robust_transforms 
    full_dataset.mask_transform = mask_transform

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Modello
    model = MiniUNet(in_channels=3, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- SCHEDULER ---
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    train_hist = []
    val_hist = []
    
    start_time = time.time()

    print("\n--- INIZIO TRAINING ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train = train_loss / len(train_loader)
        train_hist.append(avg_train)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = combined_loss(outputs, masks)
                val_loss += loss.item()
        
        avg_val = val_loss / len(val_loader)
        val_hist.append(avg_val)
        
        # Aggiorna lo scheduler
        scheduler.step(avg_val)
        
        # Lettura manuale del LR corrente per stamparlo
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train: {avg_train:.4f} | Val: {avg_val:.4f} | LR: {current_lr:.6f}")

    total_time = time.time() - start_time
    print(f"⏱️ Tempo totale: {total_time/60:.2f} minuti")

    # 3. Grafico Finale
    plt.figure(figsize=(10, 6))
    plt.plot(train_hist, label='Training Loss')
    plt.plot(val_hist, label='Validation Loss')
    plt.title('Robust Training Curves (Scheduler + Targeted Augmentation)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    os.makedirs('./outputs', exist_ok=True)
    plt.savefig('./outputs/robust_final_curves.png')
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Training concluso. Modello salvato in {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_robust()