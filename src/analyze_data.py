import torch
from torch.utils.data import DataLoader
from dataset import PetDataset
from tqdm import tqdm

def analyze_dataset():
    # Carichiamo il dataset SENZA normalizzazione (per calcolare la vera media)
    # Nota: Modifica momentaneamente dataset.py per restituire tensori puri senza Normalize
    print("Avvio analisi del dataset...")
    
    dataset = PetDataset(root_dir='./data', split='trainval')
    # Importante: batch_size=1 per analizzare immagine per immagine o accumulare stats
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Variabili per calcolo Mean/Std
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_pixels_count = 0
    
    # Variabili per bilanciamento classi (0: Background, 1: Pet, 2: Border - se presente)
    class_counts = {0: 0, 1: 0} 

    print("Calcolo statistiche pixel e distribuzione classi...")
    for img, mask in tqdm(loader):
        # img shape: [1, 3, 256, 256]
        # mask shape: [1, 256, 256]
        
        # 1. Accumulo per Mean/Std
        # Somma sui pixel (H, W) per ogni canale
        mean += img.sum(dim=(0, 2, 3)) 
        std += (img ** 2).sum(dim=(0, 2, 3))
        total_pixels_count += img.size(0) * img.size(2) * img.size(3)
        
        # 2. Conteggio Classi (Background vs Pet)
        # mask è binaria (0 o 1)
        pet_pixels = mask.sum().item()
        total_mask_pixels = mask.numel()
        bg_pixels = total_mask_pixels - pet_pixels
        
        class_counts[1] += pet_pixels
        class_counts[0] += bg_pixels

    # Calcolo finale Mean e Std
    mean /= total_pixels_count
    std = torch.sqrt((std / total_pixels_count) - (mean ** 2))

    # Calcolo percentuali classi
    total_class_pixels = class_counts[0] + class_counts[1]
    perc_bg = (class_counts[0] / total_class_pixels) * 100
    perc_pet = (class_counts[1] / total_class_pixels) * 100

    print("\n--- RISULTATI ANALISI ---")
    print(f"Mean calcolata: {mean}")
    print(f"Std calcolata: {std}")
    print("-" * 20)
    print(f"Pixel Sfondo: {class_counts[0]} ({perc_bg:.2f}%)")
    print(f"Pixel Animale: {class_counts[1]} ({perc_pet:.2f}%)")
    
    # Consiglio per la relazione:
    ratio = class_counts[0] / class_counts[1]
    print(f"Rapporto Sbilanciamento: 1 pixel Pet ogni {ratio:.2f} pixel Sfondo")

if __name__ == "__main__":
    analyze_dataset()