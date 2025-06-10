import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# ----------------------------------------
# configurare cai si hyperparametri
# ----------------------------------------
train_csv      = "train.csv"         # fisier cu date de antrenare
val_csv        = "validation.csv"    # fisier cu date de validare
test_csv       = "test.csv"          # fisier cu date de test
train_dir      = "train"             # director cu imaginile de antrenare
val_dir        = "validation"        # director cu imaginile de validare
test_dir       = "test"              # director cu imaginile de test
submission_csv = "cnn_submission.csv" # fisier unde salvam predictiile finale
best_model_pth = "best_model.pth"    # fisier unde salvam modelul cu cea mai buna validare
img_size       = (100, 100)          # dimensiunea la care redimensionam imaginile
batch_size     = 64                  # numar de imagini per batch
lr             = 1e-3                # rata de invatare
epochs         = 10                  # numar de epoci de antrenare
device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")     # afisam device-ul folosit

# ----------------------------------------
# dataset custom pentru deepfake
# ----------------------------------------
class DeepfakeDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, with_labels=True):
        self.df          = pd.read_csv(csv_file)                     # citim csv-ul intr-un dataframe
        self.img_dir     = img_dir                                   # director cu imaginile
        self.transform   = transform                                 # transformari de aplicat
        self.with_labels = with_labels                               # flag pentru etichete

    def __len__(self) -> int:
        return len(self.df)  # numar de exemple

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, 'image_id']                        # preluam id-ul imaginii
        path   = os.path.join(self.img_dir, f"{img_id}.png")         # construim calea catre fisier
        img    = Image.open(path).convert('RGB')                     # incarcam imaginea si convertim la RGB
        if self.transform:
            img = self.transform(img)                                # aplicam transformari
        if self.with_labels:
            label = int(self.df.loc[idx, 'label'])                  # preluam eticheta
            return img, label                                       # returnam imagine si eticheta
        else:
            return img, img_id                                      # returnam imagine si id pentru test

# ----------------------------------------
# transformari simple (fara augmentare)
# ----------------------------------------
transform = transforms.Compose([
    transforms.Resize(img_size),  # redimensionare la dimensiunea setata
    transforms.ToTensor()         # convertire in tensor
])

# ----------------------------------------
# creare dataloader-e
# ----------------------------------------
train_ds    = DeepfakeDataset(train_csv, train_dir,   transform, with_labels=True)
val_ds      = DeepfakeDataset(val_csv,   val_dir,     transform, with_labels=True)
test_ds     = DeepfakeDataset(test_csv,  test_dir,    transform, with_labels=False)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

# ----------------------------------------
# definire arhitectura CNN
# ----------------------------------------
class CNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # bloc de extragere a caracteristicilor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # conv initial cu 32 filtre
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # downsample la jumatate

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # conv cu 64 filtre
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)                              # downsample la jumatate
        )
        # bloc de clasificare finala
        self.classifier = nn.Sequential(
            nn.Flatten(),                                # aplatizare tensor
            nn.Linear(64*25*25, 256),                    # fully-connected 256
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),                         # fully-connected 512
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)                  # strat final cu num_classes output-uri
        )

    def forward(self, x):
        x = self.features(x)   # extragem feature-uri
        return self.classifier(x)  # obtinem predictiile

# ----------------------------------------
# initializare model, loss si optimizer
# ----------------------------------------
model     = CNN(num_classes=len(train_ds.df['label'].unique())).to(device)
criterion = nn.CrossEntropyLoss()           # functie de pierdere cross-entropy
optimizer = optim.Adam(model.parameters(), lr=lr)  # optimizator Adam

# ----------------------------------------
# functie de evaluare acuratete si pierdere
# ----------------------------------------
def evaluate(loader, compute_loss=False):
    model.eval()                             # setam model in modul evaluare
    correct, total = 0, 0
    preds, labels = [], []
    loss_sum = 0.0
    with torch.no_grad():                    # fara calcul gradienti
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)            # forward pass
            _, p = outputs.max(1)            # predictem clasa maxima
            correct += (p == lbls).sum().item()
            total   += lbls.size(0)
            preds.extend(p.cpu().numpy())    # colectam predictii
            labels.extend(lbls.cpu().numpy())# colectam etichete reale
            if compute_loss:
                loss_sum += criterion(outputs, lbls).item() * imgs.size(0)
    acc = correct / total
    if compute_loss:
        return acc, loss_sum / total, preds, labels
    return acc, preds, labels

# ----------------------------------------
# antrenare si validare
# ----------------------------------------
train_losses     = []
val_losses       = []
train_accuracies = []
val_accuracies   = []
best_val_acc     = 0.0

for ep in range(1, epochs+1):
    model.train()                           # setam modul antrenare
    running_loss = 0.0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()               # reset grad
        outputs = model(imgs)               # forward
        loss    = criterion(outputs, lbls)  # pierdere
        loss.backward()                     # backprop
        optimizer.step()                    # update parametri
        running_loss += loss.item() * imgs.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    train_acc, _, _         = evaluate(train_loader, compute_loss=False)
    train_accuracies.append(train_acc)

    val_acc, val_loss, _, _ = evaluate(val_loader, compute_loss=True)
    val_accuracies.append(val_acc)
    val_losses.append(val_loss)

    # salvam modelul daca acuratetea de validare e cea mai buna
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_pth)

    print(f"epoch {ep}/{epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - "
          f"train_acc: {train_acc:.4f} - val_acc: {val_acc:.4f}")

# ----------------------------------------
# incarcare cel mai bun model pentru predictii
# ----------------------------------------
model.load_state_dict(torch.load(best_model_pth))
model.eval()

# ----------------------------------------
# calcul matrice de confuzie pe validare
# ----------------------------------------
_, val_preds, val_labels = evaluate(val_loader)
cmatrix = confusion_matrix(val_labels, val_preds)

# ----------------------------------------
# plot evolutie pierdere si acuratete
# ----------------------------------------
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(range(1,epochs+1), train_losses, label='train loss')
plt.plot(range(1,epochs+1), val_losses,   label='val loss')
plt.xlabel('epoca'); plt.ylabel('loss'); plt.legend(); plt.title('evolutie pierdere')
plt.subplot(1,2,2)
plt.plot(range(1,epochs+1), train_accuracies, label='train acc')
plt.plot(range(1,epochs+1), val_accuracies,   label='val acc')
plt.xlabel('epoca'); plt.ylabel('acuratețe'); plt.legend(); plt.title('evolutie acuratețe')
plt.tight_layout()
plt.savefig('metrics_over_epochs.png')
plt.show()

# ----------------------------------------
# plot matrice de confuzie cu valori
# ----------------------------------------
plt.figure(figsize=(6,5))
plt.imshow(cmatrix, interpolation='nearest', cmap='Blues')
plt.title('matrice de confuzie validare')
plt.xlabel('predicted'); plt.ylabel('true')
plt.colorbar()
ticks = np.arange(cmatrix.shape[0])
plt.xticks(ticks, ticks); plt.yticks(ticks, ticks)
th = cmatrix.max()/2.
for i in range(cmatrix.shape[0]):
    for j in range(cmatrix.shape[1]):
        plt.text(j, i, cmatrix[i,j], ha='center', va='center',
                 color='white' if cmatrix[i,j]>th else 'black')
plt.tight_layout()
plt.savefig('matrice_confuzie_annotated.png')
plt.show()

# ----------------------------------------
# inferenta finala si salvare submissie
# ----------------------------------------
ids, preds = [], []
with torch.no_grad():
    for imgs, img_ids in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        _, p = outputs.max(1)
        preds.extend(p.cpu().numpy())
        ids.extend(img_ids)
submission = pd.DataFrame({'image_id': ids, 'label': preds})
submission.to_csv(submission_csv, index=False)
print(f"model cu best_val_acc={best_val_acc:.4f} salvat si submissie generata")
