import torch
from torchvision.datasets import OxfordIIITPet
import matplotlib.pyplot as plt
import os

# Configurazione
DATA_PATH = './data'

def download_and_check():
    print("Download del dataset Oxford-IIIT Pet in corso... (potrebbe richiedere qualche minuto)")
    
    # Scarica il dataset (Immagini + Maschere di segmentazione)
    dataset = OxfordIIITPet(root=DATA_PATH, 
                            split='trainval', 
                            target_types='segmentation', 
                            download=True)
    
    print(f"Dataset scaricato! Numero di campioni: {len(dataset)}")

    # Prendi un campione a caso (immagine e maschera)
    idx = 100
    image, mask = dataset[idx]

    # Visualizzazione per controllo
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title("Immagine Input")
    ax[0].axis('off')

    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("Maschera Ground Truth")
    ax[1].axis('off')

    plt.tight_layout()
    # Salva l'immagine di controllo invece di mostrarla (utile se non hai GUI)
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/data_check.png')
    print("Immagine di controllo salvata in 'outputs/data_check.png'")

if __name__ == "__main__":
    download_and_check()