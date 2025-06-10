import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# configurare cai fisier si directoare
train_csv   = "train.csv"       # fisierul cu date de antrenare
val_csv     = "validation.csv"  # fisierul cu date de validare
test_csv    = "test.csv"        # fisierul cu date de test
train_dir   = "train"           # directorul cu imaginile de antrenare
val_dir     = "validation"      # directorul cu imaginile de validare
test_dir    = "test"            # directorul cu imaginile de test

def load_flattened_images(csv_path, img_dir, with_labels=True):
    # citeste csv si obtine lista de id-uri
    df = pd.read_csv(csv_path)
    ids = df['image_id'].tolist()
    X, y = [], []
    for img_id in ids:
        # construieste calea catre imagine si o incarca in rgb
        path = os.path.join(img_dir, f"{img_id}.png")
        img = Image.open(path).convert('RGB')
        # converteste imaginea intr-un vector 1d de float32
        arr = np.asarray(img, dtype=np.float32).ravel()
        X.append(arr)
    X = np.stack(X)  # array de forma (n_samples, n_features)
    if with_labels:
        # extrage vectorul de etichete
        y = df['label'].values
        return X, y, ids
    return X, None, ids

# incarcare date de antrenare si validare
X_train, y_train, train_ids = load_flattened_images(train_csv, train_dir,   with_labels=True)
X_val,   y_val,   val_ids   = load_flattened_images(val_csv,   val_dir,     with_labels=True)

# definire pipeline: scalare + mlp
pipeline = Pipeline([
    # pas 1: standardizare caracteristici la media 0 si varianta 1
    ('scaler', StandardScaler()),
    # pas 2: retea neurala cu doua straturi ascunse
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(256, 512),  # dimensiunea straturilor ascunse
        activation='relu',              # functie de activare relu
        solver='adam',                  # algoritmul de optimizare adam
        batch_size=64,                  # marimea batch-ului
        learning_rate_init=1e-3,        # rata initiala de invatare
        max_iter=100,                   # numarul maxim de epoci
        random_state=42,
        verbose=True                    # afiseaza progresul antrenarii
    ))
])

# antrenare model mlp
pipeline.fit(X_train, y_train)

# predictie si evaluare pe setul de validare
y_pred_val = pipeline.predict(X_val)
val_acc    = accuracy_score(y_val, y_pred_val)
print(f'validation accuracy: {val_acc:.4f}')

# extragere curba de pierdere din obiectul mlp
mlp = pipeline.named_steps['mlp']
loss_curve = mlp.loss_curve_

# plot evolutie loss pe epoci
plt.figure(figsize=(8, 5))
plt.plot(loss_curve)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('training loss progression')
plt.tight_layout()
plot_path = 'training_loss_progression.png'
plt.savefig(plot_path)  # salvare grafic pe disc
plt.show()

# incarcare date de test si predictie
X_test, _, test_ids = load_flattened_images(test_csv, test_dir, with_labels=False)
y_test_pred         = pipeline.predict(X_test)

# salvare fisier submissie
submission_df = pd.DataFrame({'image_id': test_ids, 'label': y_test_pred})
submission_df.to_csv('mlp_submission.csv', index=False)
print('submission saved to mlp_submission.csv')
