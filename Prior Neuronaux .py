import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

# Configuration du device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==============================================================================
# Architecture U-Net
# ==============================================================================
class MyUNet(nn.Module):
    def __init__(self):
        super(MyUNet, self).__init__()

        # ----------------- Encoder (Downsampling) -----------------
        self.conv1 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)

        # ----------------- Decoder (Upsampling) -------------------
        self.trans1 = nn.ConvTranspose2d(256, 128, 4, padding=1, stride=2)
        self.trans2 = nn.ConvTranspose2d(256, 64, 4, padding=1, stride=2)
        self.trans3 = nn.ConvTranspose2d(128, 32, 4, padding=1, stride=2)
        self.trans4 = nn.ConvTranspose2d(32, 3, 4, padding=1, stride=2)

        # ----------------- Utils & Skips --------------------------
        self.pool = nn.MaxPool2d(2)
        # Convolutions pour aligner les dimensions des skip connections
        self.skip64 = nn.Conv2d(64, 64, 4, padding=2)
        self.skip128 = nn.Conv2d(128, 128, 5, padding=2)

    def forward(self, x):
        # --- Encodage ---
        x1_down = F.leaky_relu(self.conv1(x))  # 16 -> 32
        x1_pool = self.pool(x1_down)  # H/2

        x2_down = F.leaky_relu(self.conv2(x1_pool))  # 32 -> 64
        x2_pool = self.pool(x2_down)  # H/4
        skip64 = F.leaky_relu(self.skip64(x2_pool))

        x3_down = F.leaky_relu(self.conv3(x2_pool))  # 64 -> 128
        x3_pool = self.pool(x3_down)  # H/8
        skip128 = F.leaky_relu(self.skip128(x3_pool))

        x4_down = F.leaky_relu(self.conv4(x3_pool))  # 128 -> 256
        x4_pool = self.pool(x4_down)  # H/16

        # --- Décodage ---
        x1_up = F.leaky_relu(self.trans1(x4_pool))  # 256 -> 128

        # Fusion Skip 128
        skip128 = F.interpolate(skip128, size=x1_up.shape[2:], mode="bilinear", align_corners=False)
        x1_cat = torch.cat((x1_up, skip128), 1)

        x2_up = F.leaky_relu(self.trans2(x1_cat))  # 256 -> 64

        # Fusion Skip 64
        skip64 = F.interpolate(skip64, size=x2_up.shape[2:], mode="bilinear", align_corners=False)
        x2_cat = torch.cat((skip64, x2_up), 1)

        x3_up = F.leaky_relu(self.trans3(x2_cat))  # 128 -> 32
        x4_up = torch.sigmoid(self.trans4(x3_up))  # 32 -> 3 (RGB, [0,1])

        return x4_up


# ==============================================================================
# Préparation Globale
# ==============================================================================

try:
    input_img = Image.open('testcrop.jpg')
except FileNotFoundError:
    print("Erreur: Image 'testcrop.jpg' introuvable.")
    exit()

transform = transforms.Compose([transforms.PILToTensor()])
target = transform(input_img) / 255.0
target = target.unsqueeze(0).float().to(device)  # [1, 3, H, W]

h, w = target.size()[2], target.size()[3]

# Entrée fixe (bruit) qui sera optimisée vers l'image cible
z = torch.rand(1, 16, h, w).to(device)

criterion = nn.MSELoss()


# ==============================================================================
# Boucle d'entraînement générique
# ==============================================================================

def train_loop(name, epochs, closure_loss, save_path, perturb_z_fn=None):
    """
    Args:
        perturb_z_fn: Fonction pour ajouter du bruit à l'entrée Z à chaque itération (nécessaire pour le débruitage)
    """
    print(f"--- Démarrage : {name} ---")

    # 1. Réinitialisation complète du modèle et de l'optimiseur pour chaque tâche
    model = MyUNet().to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losslog = []

    for i in range(epochs):
        optimizer.zero_grad()

        # 2. Gestion de la perturbation de l'entrée (DIP Denoising)
        if perturb_z_fn is not None:
            input_z = perturb_z_fn(z)
        else:
            input_z = z

        output = model(input_z)
        loss = closure_loss(output)

        loss.backward()
        optimizer.step()
        losslog.append(loss.item())

        if (i + 1) % 500 == 0:
            print(f"Epoch {i + 1}/{epochs} - Loss: {loss.item():.6f}")

    # Sauvegarde résultat
    img_out = output[0].cpu().detach().permute(1, 2, 0).numpy()
    plt.imsave(save_path, img_out)

    # Affichage courbe loss
    plt.figure(figsize=(6, 4))
    plt.yscale('log')
    plt.plot(losslog, label=f'Loss finale ({losslog[-1]:.4f})')
    plt.title(f"Convergence : {name}")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
    plt.close()


# ==============================================================================
# Tâches Spécifiques
# ==============================================================================

def reconstruction():
    def loss_fn(output):
        return criterion(output, target)

    # Pas de perturbation pour la reconstruction simple
    train_loop("Reconstruction", 4000, loss_fn, 'final_reconstruction.jpg', perturb_z_fn=None)


def inpainting():
    # Création du masque (1 = garder, 0 = supprimer/trou)
    mask = torch.rand_like(target)
    mask = (mask > 0.20).float()  # 20% de pixels supprimés

    # Visualisation du masque
    plt.imshow(1 - mask[0, 0].cpu(), cmap='gray')
    plt.title("Masque (Blanc = trous)")
    plt.axis('off')
    plt.show()

    def loss_fn(output):
        # Loss calculée uniquement sur les pixels connus
        return criterion(output * mask, target * mask)

    train_loop("Inpainting", 5000, loss_fn, 'final_inpainting.jpg', perturb_z_fn=None)


def debruitage():
    # Création de l'image bruitée (cible)
    bruit = torch.randn_like(target)
    target_bruite = target + (bruit * 0.1)
    target_bruite = torch.clamp(target_bruite, 0, 1)

    plt.imshow(target_bruite[0].permute(1, 2, 0).cpu())
    plt.title("Cible Bruitée")
    plt.axis('off')
    plt.show()

    def loss_fn(output):
        return criterion(output, target_bruite)

    # ajoute du bruit à l'entrée Z à chaque étape
    # pour empêcher le réseau d'apprendre le bruit de la cible.
    def perturb_z(current_z):
        noise = torch.randn_like(current_z) * (1.0 / 30.0)
        return current_z + noise


    train_loop("Débruitage", 1500, loss_fn, 'final_debruitage.jpg', perturb_z_fn=perturb_z)


# ==============================================================================
# Exécution
# ==============================================================================

if __name__ == "__main__":
    # Décommenter la tâche souhaitée
    reconstruction()
    # inpainting()
    #debruitage()