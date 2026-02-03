import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import PetDataset
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np

# --- CONFIGURAZIONE ---
# Assicurati che punti al file giusto (quello da 25 epoche appena finito)
MODEL_PATH = './models/model_baseline.pth' 
DATA_PATH = './data'
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- DEFINIZIONE DELLA UNET ORIGINALE (Grande) ---
# La ridefiniamo qui per essere sicuri che combaci col tuo file salvato
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class StandardUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(StandardUNet, self).__init__()
        # Questa struttura corrisponde ai pesi che hai salvato (d1, d2, ecc.)
        self.d1 = DoubleConv(in_channels, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.d4 = DoubleConv(256, 512)
        
        self.bottleneck = DoubleConv(512, 1024)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.u1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.u2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.u3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.u4 = DoubleConv(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Down
        x1 = self.d1(x)
        p1 = self.pool(x1)
        x2 = self.d2(p1)
        p2 = self.pool(x2)
        x3 = self.d3(p2)
        p3 = self.pool(x3)
        x4 = self.d4(p3)
        p4 = self.pool(x4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Up
        up_1 = self.up1(b)
        # Resize se necessario (gestione dimensioni dispari)
        if up_1.shape != x4.shape:
            up_1 = nn.functional.interpolate(up_1, size=x4.shape[2:])
        merge1 = torch.cat([x4, up_1], dim=1)
        x_up_1 = self.u1(merge1)
        
        up_2 = self.up2(x_up_1)
        if up_2.shape != x3.shape:
            up_2 = nn.functional.interpolate(up_2, size=x3.shape[2:])
        merge2 = torch.cat([x3, up_2], dim=1)
        x_up_2 = self.u2(merge2)
        
        up_3 = self.up3(x_up_2)
        if up_3.shape != x2.shape:
            up_3 = nn.functional.interpolate(up_3, size=x2.shape[2:])
        merge3 = torch.cat([x2, up_3], dim=1)
        x_up_3 = self.u3(merge3)
        
        up_4 = self.up4(x_up_3)
        if up_4.shape != x1.shape:
            up_4 = nn.functional.interpolate(up_4, size=x1.shape[2:])
        merge4 = torch.cat([x1, up_4], dim=1)
        x_up_4 = self.u4(merge4)
        
        return self.final_conv(x_up_4)

def analyze_failures():
    print(f"🕵️‍♂️ Analisi Errori sul modello STANDARD: {MODEL_PATH}")
    
    # 1. Carica Modello (Usando la classe StandardUNet)
    model = StandardUNet(in_channels=3, out_channels=1).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ Modello caricato con successo!")
        model.eval()
    except Exception as e:
        print(f"❌ Errore caricamento modello: {e}")
        return

    # 2. Dataset di Test
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor()
    ])
    
    dataset = PetDataset(root_dir=DATA_PATH, split='test') 
    dataset.img_transform = transform
    dataset.mask_transform = mask_transform
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    worst_cases = []

    print("   Scansione dataset per trovare fallimenti...")
    with torch.no_grad():
        for i, (img, mask) in enumerate(loader):
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)
            
            output = model(img)
            pred = (torch.sigmoid(output) > 0.5).float()
            
            # Calcolo IoU
            intersection = (pred * mask).sum()
            union = pred.sum() + mask.sum() - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            
            # Salviamo se l'IoU è basso
            if iou < 0.4: # Soglia un po' più alta per trovare più esempi
                worst_cases.append((iou.item(), img, mask, pred))

    # Ordiniamo per errore peggiore
    worst_cases.sort(key=lambda x: x[0])
    
    # Prendiamo i peggiori 6
    top_fails = worst_cases[:6]
    print(f"   Trovati {len(worst_cases)} casi critici. Mostro i peggiori {len(top_fails)}.")

    if top_fails:
        rows = len(top_fails)
        fig, axs = plt.subplots(rows, 3, figsize=(10, 3 * rows))
        if rows == 1: axs = [axs] # Gestione caso singolo
        
        for idx, (iou, img, gt, pred) in enumerate(top_fails):
            # Denormalizza immagine
            img_disp = img.squeeze().cpu().permute(1, 2, 0).numpy()
            img_disp = img_disp * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_disp = np.clip(img_disp, 0, 1)
            
            # Se ci sono più righe usiamo axs[idx, 0], altrimenti axs[0]
            ax_row = axs[idx] if rows > 1 else axs
            
            ax_row[0].imshow(img_disp)
            ax_row[0].set_title(f"IoU: {iou:.2f}")
            ax_row[0].axis('off')
            
            ax_row[1].imshow(gt.squeeze().cpu(), cmap='gray')
            ax_row[1].set_title("Realtà")
            ax_row[1].axis('off')
            
            ax_row[2].imshow(pred.squeeze().cpu(), cmap='gray')
            ax_row[2].set_title("Predizione Errata")
            ax_row[2].axis('off')
            
        plt.tight_layout()
        plt.savefig('./outputs/failure_analysis.png')
        print("✅ Analisi salvata in outputs/failure_analysis.png")
    else:
        print("Nessun errore grave trovato sotto la soglia 0.4.")

if __name__ == "__main__":
    analyze_failures()