import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from unet import MiniUNet
from dataset import PetDataset
from torchvision import transforms
import numpy as np
import random
import os

# --- CONFIGURAZIONE ---
MODEL_PATH = './models/model_robust.pth'
DATA_PATH = './data'
SAVE_PATH = './outputs/final_report_images.png'
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def show_results():
    print(f"📸 Generazione immagini finali da {MODEL_PATH}...")
    
    # 1. Modello
    model = MiniUNet(in_channels=3, out_channels=1).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except:
        print("❌ Errore: Modello non trovato!")
        return
    model.eval()

    # 2. Dataset
    # Definiamo le trasformazioni 'pulite' per la visualizzazione
    inference_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transforms = transforms.Compose([
        transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor()
    ])

    dataset = PetDataset(root_dir=DATA_PATH, split='test') # o 'trainval'
    # Sovrascriviamo le trasformazioni del dataset per essere sicuri
    dataset.img_transform = inference_transforms
    dataset.mask_transform = mask_transforms
    
    # 3. Prendi 3 immagini a caso
    indices = random.sample(range(len(dataset)), 3)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False)

    # 4. Plot
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    
    with torch.no_grad():
        for i, (image, mask) in enumerate(loader):
            image = image.to(DEVICE)
            
            # Predizione
            output = model(image)
            pred_mask = torch.sigmoid(output) > 0.5 
            
            # Visualizzazione
            # Img: Denormalize
            img_disp = image.squeeze().cpu().permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_disp = std * img_disp + mean
            img_disp = np.clip(img_disp, 0, 1)
            
            # Maschera GT
            mask_disp = mask.squeeze().cpu().numpy()
            
            # Maschera Pred
            pred_disp = pred_mask.squeeze().cpu().numpy()

            axs[i, 0].imshow(img_disp)
            axs[i, 0].set_title("Input (128x128)")
            axs[i, 0].axis('off')
            
            axs[i, 1].imshow(mask_disp, cmap='gray')
            axs[i, 1].set_title("Ground Truth")
            axs[i, 1].axis('off')
            
            axs[i, 2].imshow(pred_disp, cmap='gray')
            axs[i, 2].set_title(f"Predizione (Robust)")
            axs[i, 2].axis('off')

    plt.tight_layout()
    os.makedirs('./outputs', exist_ok=True)
    plt.savefig(SAVE_PATH)
    print(f"✅ Immagine salvata in {SAVE_PATH}")

if __name__ == "__main__":
    show_results()