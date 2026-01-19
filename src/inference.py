import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from unet import UNet
from dataset import PetDataset
import numpy as np
import random

# --- CONFIGURAZIONE ---
MODEL_PATH = './models/model_robust.pth' # Usiamo il modello migliore
DATA_PATH = './data'
SAVE_PATH = './outputs/final_report_images.png'
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def show_results():
    # 1. Carica Modello
    print(f"Caricamento modello da {MODEL_PATH}...")
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. Carica Dataset (Modalità Validation/Test)
    dataset = PetDataset(root_dir=DATA_PATH, split='test') # Usa 'test' se esiste, o 'trainval'
    
    # 3. Prendi 3 indici casuali
    indices = random.sample(range(len(dataset)), 3)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False)

    # 4. Setup Plot
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    # Colonne: Input RGB | Ground Truth | Predizione Modello
    
    print("Generazione inferenza...")
    
    with torch.no_grad():
        for i, (image, mask) in enumerate(loader):
            image = image.to(DEVICE)
            
            # Forward Pass
            output = model(image)
            pred_mask = torch.sigmoid(output) > 0.5 # Binarizzazione
            
            # --- VISUALIZZAZIONE ---
            # Colonna 1: Immagine RGB (Denormalizza per visualizzare colori corretti)
            img_disp = image.squeeze().cpu().permute(1, 2, 0)
            img_disp = img_disp * 0.229 + 0.485     # Denormalize std & mean
            img_disp = np.clip(img_disp, 0, 1)
            
            axs[i, 0].imshow(img_disp)
            axs[i, 0].set_title("Input Image")
            axs[i, 0].axis('off')
            
            # Colonna 2: Ground Truth
            axs[i, 1].imshow(mask.squeeze(), cmap='gray')
            axs[i, 1].set_title("Ground Truth")
            axs[i, 1].axis('off')
            
            # Colonna 3: Predizione
            axs[i, 2].imshow(pred_mask.squeeze().cpu(), cmap='gray')
            axs[i, 2].set_title("Model Prediction")
            axs[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f"✅ Immagine salvata in {SAVE_PATH}")

if __name__ == "__main__":
    show_results()