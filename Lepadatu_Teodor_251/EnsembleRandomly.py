import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import optuna
import matplotlib.pyplot as plt

# ----------------------------------------
# configurare cai si hyperparametri
# ----------------------------------------
train_csv    = 'train.csv'         # fisier cu date de antrenare
val_csv      = 'validation.csv'    # fisier cu date de validare
test_csv     = 'test.csv'          # fisier cu date de test
train_dir    = 'train'             # director cu imaginile de antrenare
val_dir      = 'validation'        # director cu imaginile de validare
test_dir     = 'test'              # director cu imaginile de test
batch_size   = 64                  # numar de imagini pe batch
img_height   = 100                 # inaltimea la care redimensionam
img_width    = 100                 # latimea la care redimensionam
num_classes  = 5                   # numarul de clase
epochs       = 100                 # numar maxim de epoci per trial
n_ensemble   = 5                   # cate modele sa includem in ansamblu
max_trials   = 30                  # numar maxim de incercari optuna
max_patience = 10                  # epoci fara imbunatatire in early-stopping

# setam device-ul (cuda daca exista, altfel cpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {device}")
if torch.cuda.is_available():
    print("cuda available:", torch.cuda.is_available())
    print("cuda version:", torch.version.cuda)
    print("cudnn version:", torch.backends.cudnn.version())
    print("pytorch version:", torch.__version__)

# ----------------------------------------
# dataset custom care incarca imaginile din csv
# ----------------------------------------
class ImageCsvDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, has_label=True):
        df = pd.read_csv(csv_file)                              # citim tabelul csv
        df['filename'] = df['image_id'].astype(str) + '.png'    # construim numele fisierelor
        self.img_dir   = img_dir
        self.transform = transform
        self.has_label = has_label
        self.filenames = df['filename'].tolist()                # lista de nume de fisiere
        self.labels    = df['label'].astype(int).tolist() if has_label else None

    def __len__(self):
        return len(self.filenames)  # returnam numarul de exemple

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames[idx])  # cale catre imagine
        image = Image.open(img_path).convert('RGB')                 # incarcam si convertim la RGB
        if self.transform:
            image = self.transform(image)                           # aplicam transformari
        if self.has_label:
            return image, self.labels[idx]                          # returnam imagine si eticheta
        return image                                               # returnam doar imagine pentru test

# ----------------------------------------
# calcul medie dataset pentru normalizare
# ----------------------------------------
print("computing training set mean for featurewise centering...")
base_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),  # redimensionare constanta
    transforms.ToTensor()                        # transformare in tensor [0,1]
])
mean_loader = DataLoader(
    ImageCsvDataset(train_csv, train_dir, transform=base_transform),
    batch_size=batch_size,
    shuffle=False
)
mean = torch.zeros(3)  # initializam media pe cele 3 canale
n_samples = 0
for imgs, _ in mean_loader:
    batch_samples = imgs.size(0)
    imgs = imgs.view(batch_samples, 3, -1)    # flatten spatial
    mean += imgs.mean(2).sum(0)               # acumulam media pe batch
    n_samples += batch_samples
mean /= n_samples                             # impartim la total mostre
print(f"dataset mean: {mean}")

# ----------------------------------------
# transformari cu augmentare pentru antrenare
# ----------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.RandomAffine(
        degrees=15,              # rotatii pana la 15 grade
        translate=(0.1, 0.1),    # translatii pana la 10%
        scale=(0.9, 1.1),        # zoom intre 0.9 si 1.1
        shear=10                 # shear pana la 10 grade
    ),
    transforms.RandomHorizontalFlip(),            # flip orizontal cu p=0.5
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # jitter culoare
    transforms.ToTensor(),
    transforms.Normalize(mean, torch.ones_like(mean))      # normalizare cu media calculata
])
val_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean, torch.ones_like(mean))
])

# ----------------------------------------
# creare dataloadere pentru antrenare si validare
# ----------------------------------------
def get_dataloaders():
    train_ds = ImageCsvDataset(train_csv, train_dir, transform=train_transform)
    val_ds   = ImageCsvDataset(val_csv, val_dir,   transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader

# ----------------------------------------
# functie de construire model si optiuni pentru un trial
# ----------------------------------------
def build_model(trial):
    print(f"building model for trial {trial.number}...")
    # sugestii de hiperparametri
    l2_reg   = trial.suggest_float('l2_reg', 1e-5, 1e-3, log=True)
    dropout  = trial.suggest_float('dropout', 0.3, 0.7, step=0.1)
    lr       = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    # dict cu intervale pentru numarul de filtre per strat
    capacity = {
        1: (32, 64),
        2: (64, 128),
        3: (128, 256),
        4: (256, 512),
        5: (512, 1024)
    }

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            layers_list = []
            in_ch = 3
            # generam straturi conv folosind valorile sugerate
            for i in range(1, 6):
                min_f, max_f = capacity[i]
                out_ch = trial.suggest_int(f'conv{i}_filters', min_f, max_f, step=min_f)
                layers_list += [
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                ]
                in_ch = out_ch
            self.features = nn.Sequential(*layers_list)
            # strat fully-connected final cu numar de unitati sugerat
            dense_units = trial.suggest_int('dense_units', 1024, 2048, step=128)
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_ch, dense_units),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dense_units, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    model     = CNN()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)     # paralelizare pe mai multe GPU
    model     = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    print(model)
    return model, optimizer, criterion, scheduler

# ----------------------------------------
# functii de antrenament si evaluare pe epoca
# ----------------------------------------
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0
    correct = total = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == lbls).sum().item()
        total += lbls.size(0)
    return running_loss / total, correct / total

def eval_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0
    correct = total = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
    return running_loss / total, correct / total

# ----------------------------------------
# functie obiectiv pentru optuna
# ----------------------------------------
def objective(trial):
    train_loader, val_loader = get_dataloaders()
    model, optimizer, criterion, scheduler = build_model(trial)
    best_val_acc = 0.0
    patience = 0
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss,   val_acc   = eval_epoch(model, val_loader, criterion)
        scheduler.step()
        print(f"trial {trial.number} | epoch {epoch+1}/{epochs} -> "
              f"train acc: {train_acc:.4f} | val acc: {val_acc:.4f}")
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), f"best_model_{trial.number}.pt")
        else:
            patience += 1
        if patience >= max_patience:
            break
    return best_val_acc

# ----------------------------------------
# pornim optimizarea cu optuna
# ----------------------------------------
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler())
study.optimize(objective, n_trials=max_trials)

# ----------------------------------------
# incarcam top-n ensemble din studiile optuna
# ----------------------------------------
top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:n_ensemble]
ensemble_models = []
for t in top_trials:
    model, _, _, _ = build_model(t)
    model.load_state_dict(torch.load(f"best_model_{t.number}.pt"))
    model.eval()
    ensemble_models.append(model)

# ----------------------------------------
# inferenta ansamblu pe setul de test
# ----------------------------------------
test_ds     = ImageCsvDataset(test_csv, test_dir, transform=val_transform, has_label=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
all_probs = []
for model in ensemble_models:
    probs = []
    with torch.no_grad():
        for imgs in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs.append(torch.softmax(outputs, dim=1).cpu().numpy())  # probabilitati softmax
    all_probs.append(np.vstack(probs))
avg_probs = np.mean(np.stack(all_probs, axis=0), axis=0)  # mediem probabilitatile
final = np.argmax(avg_probs, axis=1)                      # alegem clasa cu probabilitate maxima

# ----------------------------------------
# salvare fisier de submissie
# ----------------------------------------
submission = pd.DataFrame({
    'image_id': pd.read_csv(test_csv)['image_id'],
    'label':    final
})
submission.to_csv('ensemble_submission.csv', index=False)
print('saved ensemble_submission.csv')
