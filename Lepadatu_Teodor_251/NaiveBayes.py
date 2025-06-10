import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# ----------------------------------------
# configurare cai către fișiere și directoare
# ----------------------------------------
train_csv   = 'train.csv'         # fișier csv cu date de antrenare
val_csv     = 'validation.csv'    # fișier csv cu date de validare
test_csv    = 'test.csv'          # fișier csv cu date de test
train_dir   = 'train'             # director cu imaginile de antrenare
val_dir     = 'validation'        # director cu imaginile de validare
test_dir    = 'test'              # director cu imaginile de test

def load_flat_images(csv_path, img_dir, with_labels=True):
    """
    încarcă imaginile flatten-ate și, dacă există, etichetele asociate
    """
    df = pd.read_csv(csv_path)                   # citim tabelul csv
    ids = df['image_id'].tolist()                # lista de id-uri
    X, y = [], []                                # liste pentru date și etichete

    for img_id in ids:
        # construim calea și încărcăm imaginea în RGB
        path = os.path.join(img_dir, f"{img_id}.png")
        img  = Image.open(path).convert('RGB')
        # transformăm imaginea într-un vector 1d de float32
        arr = np.asarray(img, dtype=np.float32).ravel()
        X.append(arr)                            # adăugăm vectorul în listă

    X = np.stack(X)                              # array shape=(n_samples, n_features)
    if with_labels:
        y = df['label'].values                   # obținem vectorul de etichete
        return X, y, ids
    return X, None, ids

# ----------------------------------------
# încărcare date de antrenare, validare și test
# ----------------------------------------
print("loading training, validation, test data...")
X_train, y_train, _    = load_flat_images(train_csv, train_dir, with_labels=True)
X_val,   y_val, _      = load_flat_images(val_csv,   val_dir,   with_labels=True)
X_test,  _,    test_ids= load_flat_images(test_csv,  test_dir,  with_labels=False)

# ----------------------------------------
# scalare standard a caracteristicilor
# ----------------------------------------
scaler = StandardScaler()                   # inițializăm scaler-ul
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform pe antrenare
X_val_scaled   = scaler.transform(X_val)        # transform pe validare
X_test_scaled  = scaler.transform(X_test)       # transform pe test

# ----------------------------------------
# reducere dimensionalitate prin PCA
# ----------------------------------------
pca = PCA(n_components=200, random_state=42)    # păstrăm 200 componente principale
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca   = pca.transform(X_val_scaled)
X_test_pca  = pca.transform(X_test_scaled)

# ----------------------------------------
# antrenare Gaussian Naive Bayes
# ----------------------------------------
print("training gaussian naive bayes...")
nb = GaussianNB()                              # inițializăm clasificatorul
nb.fit(X_train_pca, y_train)                   # antrenăm modelul

# ----------------------------------------
# evaluare pe setul de validare
# ----------------------------------------
print("evaluating on validation set...")
y_val_pred    = nb.predict(X_val_pca)          # predictii pe validare
val_accuracy  = accuracy_score(y_val, y_val_pred)  # calcul acuratețe
print(f"validation accuracy (nb + pca): {val_accuracy:.4f}")

# ----------------------------------------
# predictie pe setul de test și salvare submissie
# ----------------------------------------
print("predicting on test data...")
y_test_pred = nb.predict(X_test_pca)            # predictii pe test
submission  = pd.DataFrame({
    'image_id': test_ids,
    'label'   : y_test_pred
})
submission.to_csv('naive_bayes_submission.csv', index=False)  # salvăm fișierul
print("saved predictions to naive_bayes_submission.csv")
