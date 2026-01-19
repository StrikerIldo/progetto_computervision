import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torchvision import transforms
from unet import UNet
from dataset import PetDataset
from utils import combined_loss
import os
import time

# --- CONFIGURAZIONE ROBUSTA ---
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
DATA_PATH = './data'
MODEL_SAVE_PATH = './models/model_robust.pth' # Salviamo con un nome diverso!

# Definiamo le trasformazioni pesanti (Data Augmentation)
# Queste simulano i difetti che il modello troverà nel test di robustezza
robust_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    # 1. Geometria: Rotazioni e ribaltamenti
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    # 2. Colore e Luce: Simula diverse condizioni ambientali
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    # 3. Qualità: Sfocatura e Rumore
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    # Aggiunta rumore casuale
    transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train_robust():
    # 1. Setup Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Retraining Robusto su Apple MPS")
    else:
        device = torch.device("cpu")
        print("⚠️ Retraining su CPU")

    # 2. Preparazione Dati con Augmentation
    print("Caricamento dataset con Data Augmentation...")
    
    # Istanziamo il dataset base
    full_dataset = PetDataset(root_dir=DATA_PATH, split='trainval')
    
    # --- TRUCCO PER SOVRASCRIVERE LE TRASFORMAZIONI ---
    # Python ci permette di cambiare l'attributo 'img_transform' dell'istanza
    # Questo evita di dover riscrivere la classe Dataset da zero.
    full_dataset.img_transform = robust_transforms
    print(">> Trasformazioni 'Heavy' applicate al dataset di training.")

    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    # Per la validazione usiamo dati puliti (o leggermente augmentati, qui usiamo gli stessi per coerenza monitoraggio)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. Modello
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    # Opzionale: Caricare i pesi del modello baseline per velocizzare (Fine-Tuning)
    # model.load_state_dict(torch.load('./models/model_baseline.pth'))
    # Ma per l'esame è meglio ri-addestrare da zero o specificare se si fa fine-tuning.
    # Qui facciamo training da zero (scratch) per dimostrare che l'augmentation funziona.
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float('inf')
    
    print("\n--- INIZIO RETRAINING ROBUSTO ---")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Step [{batch_idx+1}] Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)

        # Validazione
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
        
        print(f"Epoch [{epoch+1}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   --> Modello Robusto salvato in {MODEL_SAVE_PATH}")

    print("Retraining completato.")

if __name__ == "__main__":
    train_robust()