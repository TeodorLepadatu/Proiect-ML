import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

# ----------------------------------------
# constante si cai catre fisiere si directoare
# ----------------------------------------
TRAIN_CSV     = 'train.csv'         # fisier csv cu datele de antrenare
VAL_CSV       = 'validation.csv'    # fisier csv cu datele de validare
TEST_CSV      = 'test.csv'          # fisier csv cu datele de test
TRAIN_DIR     = 'train'             # director cu imaginile de antrenare
VAL_DIR       = 'validation'        # director cu imaginile de validare
TEST_DIR      = 'test'              # director cu imaginile de test

BATCH_SIZE    = 64                  # dimensiunea batch-ului pentru dataloader
IMG_HEIGHT    = 100                 # inaltimea la care redimensionam imaginile
IMG_WIDTH     = 100                 # latimea la care redimensionam imaginile
NUM_CLASSES   = 5                   # numarul de clase de predictie
EPOCHS        = 400                 # numarul total de epoci de antrenare
LEARNING_RATE = 3e-4                # rata initiala de invatare (3 x 10^-4)
WEIGHT_DECAY  = 5e-5                # factorul de weight decay (5 x 10^-5)
DROPOUT_P     = 0.5                 # probabilitatea de dropout in stratul fully-connected
TOP_K         = 5                   # cate modele retinem pentru ansamblu

# alegem device-ul (gpu daca e disponibil, altfel cpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {device}")

# ----------------------------------------
# dataset personalizat pentru csv si imagini
# ----------------------------------------
class ImageCsvDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, has_label=True):
        # citim tabelul csv si construim numele fisierelor
        df = pd.read_csv(csv_file)
        df['filename'] = df['image_id'].astype(str) + '.png'
        self.img_dir   = img_dir
        self.transform = transform
        self.has_label = has_label
        self.filenames = df['filename'].tolist()
        # retinem vectorul de etichete daca exista
        if has_label:
            self.labels = df['label'].astype(int).tolist()
        else:
            self.labels = None

    def __len__(self):
        # returnam numarul total de exemple
        return len(self.filenames)

    def __getitem__(self, idx):
        # incarcam imaginea corespunzatoare indexului
        img_path = os.path.join(self.img_dir, self.filenames[idx])
        image = Image.open(img_path).convert('RGB')
        # aplicam transformari daca sunt definite
        if self.transform:
            image = self.transform(image)
        # returnam fie (imagine, eticheta), fie doar imagine
        if self.has_label:
            return image, self.labels[idx]
        else:
            return image

# ----------------------------------------
# 1) calculam media dataset-ului pentru normalizare
# ----------------------------------------
print("computing training set mean for featurewise centering...")
base_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),  # redimensionare constanta
    transforms.ToTensor()                        # transformare in tensor cu valori [0,1]
])
mean_loader = DataLoader(
    ImageCsvDataset(TRAIN_CSV, TRAIN_DIR, transform=base_transform),
    batch_size=BATCH_SIZE,
    shuffle=False
)

mean = torch.zeros(3)  # initializam tensorul de medii pe cele 3 canale
n_samples = 0
for imgs, _ in mean_loader:
    batch_samples = imgs.size(0)
    imgs = imgs.view(batch_samples, 3, -1)  # flatten spatial pe canale
    mean += imgs.mean(2).sum(0)             # acumulam media pe batch
    n_samples += batch_samples
mean /= n_samples                            # impartim la numarul total de imagini
print(f"computed dataset mean: {mean}")

# ----------------------------------------
# 2) definim transformari pentru antrenare si validare
# ----------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomAffine(               # aplicam o transformare afin aleatorie
        degrees=15,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=10
    ),
    transforms.RandomHorizontalFlip(),      # flip orizontal cu probabilitate 0.5
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # jitter de culoare
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),  # distorsionare perspectiva
    transforms.ToTensor(),
    transforms.Normalize(mean, torch.ones_like(mean)),  # normalizare folosind media calculata
    transforms.RandomErasing(p=0.25)        # stergere aleatorie de portiune din imagine
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean, torch.ones_like(mean))
])

# ----------------------------------------
# 3) construim dataloader-ele
# ----------------------------------------
def get_dataloaders():
    train_ds = ImageCsvDataset(TRAIN_CSV, TRAIN_DIR, transform=train_transform, has_label=True)
    val_ds   = ImageCsvDataset(VAL_CSV, VAL_DIR,   transform=val_transform,   has_label=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    return train_loader, val_loader

train_loader, val_loader = get_dataloaders()

# ----------------------------------------
# 4) definim structura retelei CNN
# ----------------------------------------
class CNN(nn.Module):
    def __init__(self, dropout_p=DROPOUT_P):
        super(CNN, self).__init__()
        # bloc de extragere a caracteristicilor
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        # bloc de clasificare finala
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(1024, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)      # extragem feature-map-urile
        return self.classifier(x) # obtinem logits pentru fiecare clasa

# instantiem modelul si il mutam pe device
model = CNN().to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)  # paralelizare pe multiple GPU-uri
model = model.to(device)

# ----------------------------------------
# 5) definim functia de pierdere, optimizer-ul si scheduler-ul
# ----------------------------------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # cross-entropy cu label smoothing
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, eps=1e-8)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,          # epoci pana la primul restart
    T_mult=2,        # factor de inmultire a perioadei la fiecare restart
    eta_min=1e-8     # rata minima de invatare
)

# ----------------------------------------
# 6) bucla de antrenare cu salvare top-k modele
# ----------------------------------------
top_models = []  # lista de dict-uri cu cele mai bune modele

for epoch in range(1, EPOCHS + 1):
    # mod antrenare
    model.train()
    total_loss = correct = total = 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out = model(imgs)                           # forward pass
        loss = criterion(out, lbls)                 # calcul pierdere
        loss.backward()                             # backprop
        optimizer.step()                            # update parametri
        scheduler.step()                            # update lr per pas

        total_loss += loss.item() * imgs.size(0)
        preds = out.argmax(1)
        correct += (preds == lbls).sum().item()
        total += lbls.size(0)
    train_acc = correct / total

    # mod validare
    model.eval()
    v_loss = v_correct = v_total = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            loss = criterion(out, lbls)

            v_loss += loss.item() * imgs.size(0)
            preds = out.argmax(1)
            v_correct += (preds == lbls).sum().item()
            v_total += lbls.size(0)
    val_acc = v_correct / v_total

    # salvam modelul daca e in top-k dupa acuratete
    ckpt_path = f"model_epoch_{epoch:03d}.pth"
    if len(top_models) < TOP_K or val_acc > min(m['acc'] for m in top_models):
        torch.save(model.state_dict(), ckpt_path)
        top_models.append({'acc': val_acc, 'epoch': epoch, 'path': ckpt_path})
        # pastram doar cele mai bune TOP_K
        top_models = sorted(top_models, key=lambda x: x['acc'], reverse=True)[:TOP_K]

    print(f"epoch {epoch}/{EPOCHS}: train acc={train_acc:.4f}, val acc={val_acc:.4f}")

# retinem calea modelului cu cea mai mare acuratete pentru tie-break
best_model_path = top_models[0]['path']
print("top models:")
for m in top_models:
    print(f"epoch {m['epoch']}: acc={m['acc']:.4f} -> {m['path']}")

# ----------------------------------------
# 7) inferenta ansamblu pe setul de test
# ----------------------------------------
sample_sub = pd.read_csv('sample_submission.csv')  # incarc fisierul de sample
test_ds = ImageCsvDataset(TEST_CSV, TEST_DIR, transform=val_transform, has_label=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# incarcam fiecare model din ansamblu
ensemble_models = []
for m in top_models:
    net = CNN().to(device)
    net.load_state_dict(torch.load(m['path'], map_location=device))
    net.eval()
    ensemble_models.append(net)

# facem predictii prin vot majoritar
total_preds = []
with torch.no_grad():
    for imgs in test_loader:
        imgs = imgs.to(device)
        preds = [net(imgs).argmax(1).cpu().numpy() for net in ensemble_models]
        stacked = np.stack(preds, axis=1)  # shape: (batch, TOP_K)
        for row in stacked:
            counts = np.bincount(row, minlength=NUM_CLASSES)
            winners = np.where(counts == counts.max())[0]
            # daca e egalitate, luam votul primului model (cel mai bun)
            total_preds.append(winners[0] if len(winners)==1 else row[0])

# salvam fisierul de submisie final
submission = pd.DataFrame({
    'image_id': sample_sub['image_id'],
    'label':    total_preds
})
submission.to_csv('final_ensemble_submission.csv', index=False)
print("ensemble submission saved to final_ensemble_submission.csv")
