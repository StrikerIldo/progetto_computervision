import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from unet import MiniUNet
from dataset import PetDataset

# --- CONFIGURAZIONE ---
DATA_PATH = './data'
MODEL_PATH = './models/model_robust.pth' 
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 1

# --- DEFINIZIONE CORRUZIONI (Stesse usate nel training) ---
# Nota: Usiamo 128x128 per coerenza con la MiniUNet
clean_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

corrupted_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    # Applichiamo le corruzioni per testare la robustezza
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
    transforms.GaussianBlur(kernel_size=5, sigma=1.5),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x + 0.1 * torch.randn_like(x)), # Rumore
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.PILToTensor() # Restituisce 0, 1, 2
])

def calculate_iou(pred, target):
    # Pred è già sigmoidata e binarizzata (0 o 1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    if union == 0:
        return 1.0
    return intersection / union

def test_robustness():
    print(f"🧐 Avvio Test Robustezza usando MiniUNet su {DEVICE}...")
    
    # 1. Carica il Modello
    model = MiniUNet(in_channels=3, out_channels=1).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ Modello Robusto caricato correttamente.")
    except FileNotFoundError:
        print(f"❌ Errore: Non trovo il file {MODEL_PATH}. Controlla il nome.")
        return

    model.eval()

    # 2. Prepara i Dataset (Clean vs Corrupted)
    # Usiamo 'test' se esiste, altrimenti una parte di 'trainval'
    dataset_clean = PetDataset(root_dir=DATA_PATH, split='trainval')
    dataset_clean.img_transform = clean_transform
    dataset_clean.mask_transform = mask_transform
    
    dataset_corr = PetDataset(root_dir=DATA_PATH, split='trainval')
    dataset_corr.img_transform = corrupted_transform # Qui la magia del test
    dataset_corr.mask_transform = mask_transform

    # Limitiamo il test a 100 immagini per velocità
    indices = list(range(100))
    subset_clean = torch.utils.data.Subset(dataset_clean, indices)
    subset_corr = torch.utils.data.Subset(dataset_corr, indices)

    loader_clean = DataLoader(subset_clean, batch_size=BATCH_SIZE, shuffle=False)
    loader_corr = DataLoader(subset_corr, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Loop di Valutazione
    iou_clean_list = []
    iou_corr_list = []

    print("   Calcolo IoU su dati PULITI...")
    with torch.no_grad():
        for img, mask in tqdm(loader_clean):
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)
            # Normalizziamo la maschera a 0/1 (Pet/Sfondo)
            mask = (mask == 1).float() 
            
            output = model(img)
            pred = (torch.sigmoid(output) > 0.5).float()
            iou_clean_list.append(calculate_iou(pred, mask).item())

    print("   Calcolo IoU su dati CORROTTI (Domain Shift)...")
    with torch.no_grad():
        for img, mask in tqdm(loader_corr):
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)
            mask = (mask == 1).float()
            
            output = model(img)
            pred = (torch.sigmoid(output) > 0.5).float()
            iou_corr_list.append(calculate_iou(pred, mask).item())

    # 4. Risultati Finali
    mean_iou_clean = np.mean(iou_clean_list)
    mean_iou_corr = np.mean(iou_corr_list)
    drop = ((mean_iou_clean - mean_iou_corr) / mean_iou_clean) * 100

    print("\n" + "="*30)
    print("RISULTATI FINALI ROBUSTEZZA")
    print("="*30)
    print(f"Modello: {MODEL_PATH}")
    print(f"IoU Medio (Clean):     {mean_iou_clean:.4f}")
    print(f"IoU Medio (Corrupted): {mean_iou_corr:.4f}")
    print(f"Performance Drop:      {drop:.2f}%")
    print("="*30)

    if mean_iou_corr > 0.30:
        print("✅ SUCCESSO: Il modello ha recuperato la capacità di vedere nel rumore!")
    else:
        print("⚠️ ATTENZIONE: IoU ancora basso. Forse serve più training.")

if __name__ == "__main__":
    test_robustness()