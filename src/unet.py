import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    Ogni step della U-Net consiste in due convoluzioni 3x3.
    Aggiungiamo Batch Normalization e ReLU dopo ogni convoluzione.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # Padding=1 mantiene le dimensioni spaziali (H, W) invariate
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1): # Output 1 canale per maschera binaria
        super(UNet, self).__init__()
        
        # --- ENCODER (Downsampling) ---
        # Aumentiamo le feature maps (64 -> 128 -> 256 -> 512)
        self.d1 = DoubleConv(in_channels, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.d4 = DoubleConv(256, 512)
        
        # MaxPool riduce le dimensioni spaziali di 2x
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- BOTTLENECK ---
        self.bottleneck = DoubleConv(512, 1024)

        # --- DECODER (Upsampling) ---
        # Usiamo ConvTranspose2d per raddoppiare le dimensioni spaziali
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.u1 = DoubleConv(1024, 512) # Input 1024 perché concatena 512 (dal basso) + 512 (skip conn)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.u2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.u3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.u4 = DoubleConv(128, 64)

        # --- FINAL LAYER ---
        # Convoluzione 1x1 per mappare le feature maps al numero di classi (1)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        c1 = self.d1(x)
        p1 = self.pool(c1)
        
        c2 = self.d2(p1)
        p2 = self.pool(c2)
        
        c3 = self.d3(p2)
        p3 = self.pool(c3)
        
        c4 = self.d4(p3)
        p4 = self.pool(c4)
        
        # --- Bottleneck ---
        bn = self.bottleneck(p4)
        
        # --- Decoder con Skip Connections ---
        
        # Blocco 1
        up_1 = self.up1(bn)
        # Concatenazione sull'asse dei canali (dim=1). Skip connection da c4
        concat_1 = torch.cat((c4, up_1), dim=1) 
        x = self.u1(concat_1)
        
        # Blocco 2
        up_2 = self.up2(x)
        concat_2 = torch.cat((c3, up_2), dim=1) # Skip connection da c3
        x = self.u2(concat_2)
        
        # Blocco 3
        up_3 = self.up3(x)
        concat_3 = torch.cat((c2, up_3), dim=1) # Skip connection da c2
        x = self.u3(concat_3)
        
        # Blocco 4
        up_4 = self.up4(x)
        concat_4 = torch.cat((c1, up_4), dim=1) # Skip connection da c1
        x = self.u4(concat_4)
        
        # Output finale
        out = self.final_conv(x)
        
        # Nota: Non usiamo Sigmoid qui perché useremo BCEWithLogitsLoss che è più stabile
        return out

# --- Blocco di test per verificare che funzioni ---
if __name__ == "__main__":
    # Simula un'immagine batch=1, canali=3, altezza=256, larghezza=256
    x = torch.randn(1, 3, 256, 256)
    model = UNet(in_channels=3, out_channels=1)
    preds = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")
    
    # Verifica dimensionale
    assert x.shape[2:] == preds.shape[2:], "Errore: Input e Output hanno dimensioni diverse!"
    print("Test superato: Le dimensioni coincidono. U-Net pronta.")