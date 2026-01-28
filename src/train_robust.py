import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torchvision import transforms
from unet import MiniUNet  # <--- USIAMO LA STESSA RETE DEL BASELINE!
from dataset import PetDataset
from utils import combined_loss
import os
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE ---
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
DATA_PATH = './data'
MODEL_SAVE_PATH = './models/model_robust.pth'

# --- DATA AUGMENTATION ---
# Questa è la "cura" per l'overfitting che hai visto nel grafico
robust_transforms = transforms.Compose([
    transforms.Resize((128, 128)), # Manteniamo 128x128 per velocità
    # 1. Geometria: Rotazioni e ribaltamenti per invarianza posizionale
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    # 2. Colore: Per evitare che impari "quel" marrone specifico
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    # 3. Qualità: Sfocatura per robustezza ai bordi
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    # 4. Rumore: Regolarizzatore forte
    transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),
    # 5. Normalizzazione standard di ImageNet
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train_robust():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🛡️  Avvio Training Robusto (Soluzione all'Overfitting) su {device}...")

    # 1. Dataset: Sovrascriviamo le trasformazioni
    full_dataset = PetDataset(root_dir=DATA_PATH, split='trainval')
    full_dataset.img_transform = robust_transforms # Iniettiamo l'augmentation
    print("   ✅ Applicata Heavy Data Augmentation.")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # num_workers=0 per stabilità
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Modello: MiniUNet (lo stesso usato nel baseline per confronto ottimale)
    model = MiniUNet(in_channels=3, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_loss_hist = []
    val_loss_hist = []
    
    print("\n--- INIZIO TRAINING ROBUSTO ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_train = epoch_loss / len(train_loader)
        train_loss_hist.append(avg_train)

        # Validazione
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = combined_loss(outputs, masks)
                val_loss += loss.item()
        
        avg_val = val_loss / len(val_loader)
        val_loss_hist.append(avg_val)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train: {avg_train:.4f} | Val: {avg_val:.4f}")

    # 3. Salvataggio Grafico Confronto
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_hist, label='Train (Robust)')
    plt.plot(val_loss_hist, label='Val (Robust)')
    plt.title('Curve di Apprendimento con Data Augmentation')
    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./outputs/robust_learning_curves.png')
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("✅ Training completato. Grafico salvato.")

if __name__ == "__main__":
    train_robust()