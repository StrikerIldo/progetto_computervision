import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from unet import MiniUNet
from dataset import PetDataset
from utils import combined_loss
import matplotlib.pyplot as plt
import os
import time

# --- CONFIGURAZIONE PER IL PROF ---
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
DATA_PATH = './data'

def train_and_analyze():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Avvio Training (MiniUNet) su {device}...")

    # 1. Dataset
    full_dataset = PetDataset(root_dir=DATA_PATH, split='trainval')
    # Split 80/20 per avere un buon validation set
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # num_workers=0 per stabilità su Mac M1/M2
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Modello Mini (Leggero)
    model = MiniUNet(in_channels=3, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Liste per salvare i dati dei grafici
    train_loss_history = []
    val_loss_history = []
    
    print("Inizio Loop...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        # Training Step
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = combined_loss(outputs, masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 4. Generazione Grafico e Salvataggio
    print("\n📊 Generazione curve di analisi...")
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Training Loss', marker='.')
    plt.plot(val_loss_history, label='Validation Loss', marker='.')
    plt.title('Learning Curves: Overfitting Analysis')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    os.makedirs('./outputs', exist_ok=True)
    plt.savefig('./outputs/learning_curves.png')
    torch.save(model.state_dict(), './models/mini_unet.pth')
    print("✅ Fatto! Grafico salvato in outputs/learning_curves.png")

if __name__ == "__main__":
    train_and_analyze()