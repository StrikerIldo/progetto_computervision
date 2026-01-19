import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from unet import UNet
from dataset import PetDataset
from utils import combined_loss
import os
import time

# --- CONFIGURAZIONE ---
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
DATA_PATH = './data'
MODEL_SAVE_PATH = './models/model_baseline.pth'

def train():
    # 1. Setup Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Training su Apple M1/M2 (MPS)")
    else:
        device = torch.device("cpu")
        print("⚠️ Training su CPU")

    # 2. Preparazione Dati
    print("⏳ Caricamento dataset (potrebbe richiedere qualche secondo)...")
    full_dataset = PetDataset(root_dir=DATA_PATH, split='trainval')
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"✅ Dati pronti: {len(train_dataset)} immagini di training.")

    # 3. Modello
    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float('inf')
    
    os.makedirs('./models', exist_ok=True)

    print("\n--- INIZIO TRAINING (Premi Ctrl+C per interrompere) ---")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        # --- LOOP DI TRAINING ---
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # STAMPA OGNI 10 BATCH (Feedback immediato)
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)

        # --- VALIDAZIONE ---
        print(f"Validazione Epoch {epoch+1} in corso...")
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = combined_loss(outputs, masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f" >>> RISULTATO EPOCH {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f" 💾 Modello salvato (Miglioramento!)")
        
        print("-" * 50)

    total_time = time.time() - start_time
    print(f"Training finito in {total_time/60:.1f} minuti.")

if __name__ == "__main__":
    train()