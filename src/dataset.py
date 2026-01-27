import torch
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from PIL import Image
import numpy as np

class PetDataset(Dataset):
    def __init__(self, root_dir, split='trainval'):
        """
        Wrapper attorno al dataset OxfordIIITPet per gestire le trasformazioni.
        split: 'trainval' per training, 'test' per validation/test.
        """
        self.base_dataset = OxfordIIITPet(root=root_dir, split=split, target_types='segmentation', download=False)
        
        # Trasformazioni per l'immagine di input (RGB)
        # 1. Resize a 256x256
        # 2. Converti a Tensore
        # 3. Normalizza (media e deviazione standard standard di ImageNet)
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[-0.0293, -0.0453, -0.0458], std=[1.1496, 1.1509, 1.1818])
        ])

        # Trasformazioni per la maschera (Target)
        # Resize con INTER_NEAREST per non introdurre valori grigi (vogliamo solo classi intere)
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.PILToTensor() 
        ])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, mask = self.base_dataset[idx]
        
        # Applica trasformazioni
        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        
        # --- PRE-PROCESSING MASCHERA SPECIFICO PER OXFORD PET ---
        # Le maschere originali hanno 3 valori:
        # 1: Pet (Foreground)
        # 2: Sfondo (Background)
        # 3: Bordo (Outline)
        # Noi vogliamo una maschera binaria: 1 dove c'è il pet, 0 altrove.
        
        # Sottraiamo 1 per avere range 0-2 (0=Pet, 1=Sfondo, 2=Bordo)
        mask = mask - 1
        
        # Mettiamo a 1 dove c'è il pet (che ora è 0) e 0 tutto il resto
        # Nota: torch.where(condition, x, y)
        mask = torch.where(mask == 0, 1.0, 0.0)
        
        return image, mask