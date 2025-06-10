import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # import pentru matrice de confuzie

# ----------------------------------------
# constante si cai catre fisiere/directoare
# ----------------------------------------
train_csv     = 'train.csv'         # fisier cu datele de antrenare
val_csv       = 'validation.csv'    # fisier cu datele de validare
test_csv      = 'test.csv'          # fisier cu datele de test
train_dir     = 'train'             # director cu imaginile de antrenare
val_dir       = 'validation'        # director cu imaginile de validare
test_dir      = 'test'              # director cu imaginile de test

batch_size    = 64                  # numar de imagini procesate o data
img_height    = 100                 # inaltimea imaginilor dupa redimensionare
img_width     = 100                 # latimea imaginilor dupa redimensionare
num_classes   = 5                   # numar de clase de predictie
epochs        = 300                 # numar de epoci de antrenare
learning_rate = 1e-3                # rata de invatare initiala
weight_decay  = 1e-4                # factorul de regularizare l2
dropout_p     = 0.5                 # probabilitatea dropout in stratul fully-connected

# configurare device (cuda daca exista, altfel cpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using device: {device}")
if torch.cuda.is_available():
    print("cuda available:", torch.cuda.is_available())
    print("cuda version:", torch.version.cuda)
    print("cudnn version:", torch.backends.cudnn.version())
    print("pytorch version:", torch.__version__)

# ----------------------------------------
# dataset personalizat pentru csv + imagini
# ----------------------------------------
class ImageCsvDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, has_label=True):
        # citim csv-ul si adaugam extensia .png la fiecare image_id
        df = pd.read_csv(csv_file)
        df['filename'] = df['image_id'].astype(str) + '.png'
        self.img_dir   = img_dir
        self.transform = transform
        self.has_label = has_label
        self.filenames = df['filename'].tolist()
        # retinem si vectorul de etichete daca exista
        if has_label:
            self.labels = df['label'].astype(int).tolist()
        else:
            self.labels = None

    def __len__(self):
        # returneaza numarul de mostre
        return len(self.filenames)

    def __getitem__(self, idx):
        # construim calea catre imagine si o incarcam
        img_path = os.path.join(self.img_dir, self.filenames[idx])
        image = Image.open(img_path).convert('RGB')
        # aplicam transformari daca sunt definite
        if self.transform:
            image = self.transform(image)
        # returnam tuplu (imagine, eticheta) sau doar imagine
        if self.has_label:
            return image, self.labels[idx]
        else:
            return image

# ----------------------------------------
# 1) calculul mediei pe fiecare canal pentru normalizare
# ----------------------------------------
print("computing training set mean for featurewise centering...")
base_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),  # redimensionare
    transforms.ToTensor()                        # transformare in tensor [0,1]
])
mean_loader = DataLoader(
    ImageCsvDataset(train_csv, train_dir, transform=base_transform),
    batch_size=batch_size,
    shuffle=False
)

# acumulam media per canal
mean = torch.zeros(3)
n_samples = 0
for imgs, _ in mean_loader:
    batch_samples = imgs.size(0)
    imgs = imgs.view(batch_samples, 3, -1)   # flatten spatial pe fiecare canal
    mean += imgs.mean(2).sum(0)              # sumam mediile pe batch
    n_samples += batch_samples
mean /= n_samples                           # impartim la total mostre
print(f"computed dataset mean: {mean}")

# ----------------------------------------
# 2) definirea transformărilor
# ----------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.RandomAffine(                   # transformare afină aleatorie
        degrees=15,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=10
    ),
    transforms.RandomHorizontalFlip(),         # flip orizontal cu p=0.5
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # jitter culoare
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),  # perspectivă
    transforms.ToTensor(),                     # la tensor
    transforms.Normalize(mean, torch.ones_like(mean)),  # normalizare cu media calculata
    transforms.RandomErasing(p=0.25)           # stergere aleatorie dreptunghi
])

val_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean, torch.ones_like(mean))
])

# ----------------------------------------
# 3) creare dataloaders
# ----------------------------------------
def get_dataloaders():
    train_ds = ImageCsvDataset(train_csv, train_dir, transform=train_transform, has_label=True)
    val_ds   = ImageCsvDataset(val_csv, val_dir, transform=val_transform, has_label=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, val_loader

train_loader, val_loader = get_dataloaders()

# ----------------------------------------
# 4) definirea modelului CNN
# ----------------------------------------
class CNN(nn.Module):
    def __init__(self, dropout_p=dropout_p):
        super(CNN, self).__init__()
        # bloc de extragere a caracteristicilor
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # bloc de clasificare finala
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),  # pool adaptiv la 1x1
            nn.Flatten(),                         # flatten la vector
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        # fluxul forward: features -> classifier
        x = self.features(x)
        x = self.classifier(x)
        return x

# instantiere model si mutare pe device
model = CNN(dropout_p=dropout_p)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)  # paralelizare pe mai multe GPU-uri daca exista
model = model.to(device)
print(model)

# ----------------------------------------
# 5) definirea pierderii, optimizer-ului si scheduler-ului
# ----------------------------------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # cross-entropy cu label smoothing
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-8)  # adamw
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs,     # perioada totala de cosine annealing
    eta_min=1e-6      # rata minima
)

# ----------------------------------------
# 6) functii de antrenare si evaluare pe o epoca
# ----------------------------------------
def train_epoch(model, loader, optimizer, criterion):
    model.train()      # modul antrenare
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()            # reset gradienti
        outputs = model(imgs)            # predictii
        loss = criterion(outputs, lbls)  # calcul pierdere
        loss.backward()                  # backprop
        optimizer.step()                 # update parametri

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == lbls).sum().item()
        total += lbls.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc

def eval_epoch(model, loader, criterion):
    model.eval()       # modul evaluare
    running_loss = 0.0
    correct = 0
    total   = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, lbls)

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc

# ----------------------------------------
# 7) bucla principala de antrenare si logare metrici
# ----------------------------------------
train_losses = []
val_losses   = []
train_accs   = []
val_accs     = []

best_val_acc = 0.0
best_epoch   = -1

for epoch in range(1, epochs + 1):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    val_loss,   val_acc   = eval_epoch(model, val_loader, criterion)

    scheduler.step()  # actualizare rata de invatare

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    # afisare progres
    print(f"epoch [{epoch}/{epochs}]  "
          f"train loss: {train_loss:.4f} | train acc: {train_acc:.4f}  "
          f"val loss: {val_loss:.4f}   | val acc: {val_acc:.4f}")

    # salvare model daca performanta de validare e cea mai buna
    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        best_epoch   = epoch
        torch.save(model.state_dict(), "best_fixed_cnn_first.pth")

print(f"\ntraining complete. best val acc: {best_val_acc:.4f} at epoch {best_epoch}.")

# ----------------------------------------
# 8) matrice de confuzie pe setul de validare
# ----------------------------------------
model.load_state_dict(torch.load("best_fixed_cnn_first.pth", map_location=device))
model.eval()

all_preds = []
all_labels = []
with torch.no_grad():
    for imgs, lbls in val_loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(lbls)

all_preds  = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()
cm = confusion_matrix(all_labels, all_preds)  # calcul matrice de confuzie
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(num_classes)))
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax)                             # afisare matrice de confuzie
plt.title(f'validation confusion matrix (epoch {best_epoch})')
plt.savefig("confusion_matrix_1.png")
plt.show()

# ----------------------------------------
# 9) plotarea metricilor de antrenare si validare
# ----------------------------------------
epochs_range = range(1, epochs + 1)

# acuratete
plt.figure(figsize=(8, 5))
plt.plot(train_accs, label='train accuracy')
plt.plot(val_accs,   label='validation accuracy')
plt.title('accuracy per epoch')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.grid(True)
plt.savefig("acc_plot.png")
plt.show()

# loss
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='train loss')
plt.plot(val_losses,   label='validation loss')
plt.title('loss per epoch')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.grid(True)
plt.savefig("loss_plot.png")
plt.show()

# ----------------------------------------
# 10) inferenta finala si generare submission
# ----------------------------------------
test_ds = ImageCsvDataset(test_csv, test_dir, transform=val_transform, has_label=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

model.load_state_dict(torch.load("best_fixed_cnn_first.pth"))  # incarcare model final
model.eval()
predictions = []
with torch.no_grad():
    for imgs in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(1).cpu().numpy()
        predictions.extend(preds)

sample_submission = pd.read_csv("sample_submission.csv")
submission_df = pd.DataFrame({
    'image_id': sample_submission['image_id'],
    'label':    predictions
})
submission_df.to_csv("final_submission_first.csv", index=False)
print("submission file saved as: final_submission_first.csv")
