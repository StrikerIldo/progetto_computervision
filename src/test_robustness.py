import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from unet import UNet
from dataset import PetDataset
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURAZIONE ---
DATA_PATH = './data'
MODEL_PATH = './models/model_robust.pth'
OUTPUT_DIR = './outputs/robustness_test_SUCCESS'

def calculate_iou(pred_mask, true_mask):
    """Calcola l'Intersection over Union (IoU) per una singola coppia."""
    # Sigmoid e soglia a 0.5 per binarizzare
    pred_mask = torch.sigmoid(pred_mask) > 0.5
    true_mask = true_mask > 0.5
    
    intersection = (pred_mask & true_mask).sum().item()
    union = (pred_mask | true_mask).sum().item()
    
    if union == 0: return 1.0
    return intersection / union

def get_corrupted_transforms():
    """
    Definisce le trasformazioni 'cattive' per simulare il Domain Shift.
    Nota: Applichiamo queste trasformazioni SOLO alle immagini, non alle maschere!
    """
    base_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Aggiungiamo rumore e cambi di colore
    corruption_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        # ColorJitter simula condizioni di luce pessime
        transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
        # GaussianBlur simula foto sfocate
        transforms.GaussianBlur(kernel_size=5, sigma=(2.0, 5.0)),
        transforms.ToTensor(),
        # Aggiunta manuale di rumore Gaussiano
        transforms.Lambda(lambda x: x + 0.1 * torch.randn_like(x)),
        # Clamp per tenere i valori validi prima della normalizzazione
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return base_transform, corruption_transform

def test():
    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Caricamento modello...")
    model = UNet(in_channels=3, out_channels=1).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ Modello caricato con successo.")
    except FileNotFoundError:
        print(f"❌ Errore: Il file {MODEL_PATH} non esiste. Attendi la fine del training!")
        return

    model.eval()
    
    # Prepariamo due dataset: uno PULITO e uno SPORCO (Corrupted)
    clean_transform, dirty_transform = get_corrupted_transforms()
    
    # Dataset di test (usiamo 'test' split se disponibile, altrimenti 'trainval' parziale)
    # Nota: Stiamo forzando la trasformazione sovrascrivendo l'attributo nel dataset wrapper
    dataset = PetDataset(root_dir=DATA_PATH, split='trainval')
    
    # Limitiamo a 100 immagini per il test rapido
    indices = list(range(100))
    subset = torch.utils.data.Subset(dataset, indices)
    test_loader = DataLoader(subset, batch_size=1, shuffle=False)
    
    clean_ious = []
    dirty_ious = []
    
    print("\n--- INIZIO TEST DI ROBUSTEZZA ---")
    
    with torch.no_grad():
        for idx, (image, mask) in enumerate(test_loader):
            mask = mask.to(device)
            
            # --- TEST 1: IMMAGINE PULITA ---
            # Reimpostiamo la trasformazione pulita 'al volo' (un po' hacky ma efficace per il test)
            # Nota: Poiché il dataloader ha già trasformato l'immagine, qui stiamo facendo un trucco.
            # Per farlo bene dovremmo ricaricare l'immagine originale, ma per semplicità
            # applichiamo il rumore direttamente al tensore o usiamo due dataset distinti.
            #
            # CORREZIONE PER SEMPLICITÀ:
            # Poiché PetDataset applica le trasformazioni nel __getitem__, 
            # il modo più pulito è istanziare due dataset diversi.
            pass 

    # --- RISCRIVO LA LOGICA DEL LOOP PER CHIAREZZA ---
    # Per fare un confronto onesto, carichiamo l'immagine Raw e applichiamo le due pipeline
    raw_dataset = dataset.base_dataset # Accesso al dataset originale torchvision
    
    for i in range(20): # Testiamo su 20 immagini
        raw_img, raw_mask = raw_dataset[i]
        
        # 1. Preparazione Ground Truth
        mask_tensor = dataset.mask_transform(raw_mask)
        mask_tensor = mask_tensor - 1
        mask_tensor = torch.where(mask_tensor == 0, 1.0, 0.0).unsqueeze(0).to(device) # Add batch dim

        # 2. Pipeline PULITA
        img_clean = clean_transform(raw_img).unsqueeze(0).to(device)
        out_clean = model(img_clean)
        iou_clean = calculate_iou(out_clean, mask_tensor)
        clean_ious.append(iou_clean)
        
        # 3. Pipeline SPORCA (Domain Shift)
        img_dirty = dirty_transform(raw_img).unsqueeze(0).to(device)
        out_dirty = model(img_dirty)
        iou_dirty = calculate_iou(out_dirty, mask_tensor)
        dirty_ious.append(iou_dirty)
        
        # Salva un esempio visivo (solo il primo)
        if i == 0:
            fig, ax = plt.subplots(2, 3, figsize=(12, 8))
            
            # Riga 1: Pulito
            # Denormalizziamo per visualizzare
            disp_clean = img_clean.squeeze().cpu().permute(1, 2, 0) * 0.229 + 0.485
            ax[0,0].imshow(np.clip(disp_clean, 0, 1))
            ax[0,0].set_title("Input Pulito")
            ax[0,1].imshow(torch.sigmoid(out_clean).squeeze().cpu() > 0.5, cmap='gray')
            ax[0,1].set_title(f"Predizione (IoU: {iou_clean:.2f})")
            ax[0,2].imshow(mask_tensor.squeeze().cpu(), cmap='gray')
            ax[0,2].set_title("Ground Truth")
            
            # Riga 2: Sporco
            disp_dirty = img_dirty.squeeze().cpu().permute(1, 2, 0) * 0.229 + 0.485
            ax[1,0].imshow(np.clip(disp_dirty, 0, 1))
            ax[1,0].set_title("Input Corrotto (Domain Shift)")
            ax[1,1].imshow(torch.sigmoid(out_dirty).squeeze().cpu() > 0.5, cmap='gray')
            ax[1,1].set_title(f"Predizione (IoU: {iou_dirty:.2f})")
            ax[1,2].imshow(mask_tensor.squeeze().cpu(), cmap='gray')
            ax[1,2].set_title("Ground Truth")
            
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/comparison.png")
            print(f"Salvato esempio visivo in {OUTPUT_DIR}/comparison.png")

    avg_clean = sum(clean_ious) / len(clean_ious)
    avg_dirty = sum(dirty_ious) / len(dirty_ious)
    
    print("\n--- RISULTATI ---")
    print(f"IoU Medio (Immagini Pulite): {avg_clean:.4f}")
    print(f"IoU Medio (Immagini Corrotte): {avg_dirty:.4f}")
    print(f"DROP DI PERFORMANCE: -{(avg_clean - avg_dirty)*100:.2f}%")
    
    # Scrivi un piccolo report txt
    with open(f"{OUTPUT_DIR}/results.txt", "w") as f:
        f.write(f"Baseline Clean IoU: {avg_clean:.4f}\n")
        f.write(f"Corrupted IoU: {avg_dirty:.4f}\n")
        f.write(f"Drop: {(avg_clean - avg_dirty)*100:.2f}%\n")

if __name__ == "__main__":
    test()