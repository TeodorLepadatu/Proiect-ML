import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ----------------------------------------
# configurare cai catre fisiere si directoare
# ----------------------------------------
TRAIN_CSV = "train.csv"           # fisier csv cu datele de antrenare
VAL_CSV   = "validation.csv"      # fisier csv cu datele de validare
TEST_CSV  = "test.csv"            # fisier csv cu datele de test
TRAIN_DIR = "train"               # director cu imaginile de antrenare
VAL_DIR   = "validation"          # director cu imaginile de validare
TEST_DIR  = "test"                # director cu imaginile de test

def load_flattened_images(csv_path, img_dir, with_labels=True):
    """
    incarca imaginile, le converteste in vectori 1d si, daca e cazul, returneaza etichete
    fara diacritice, folosind numpy pentru stocare eficienta.
    """
    # citim tabelul csv si extragem lista de id-uri
    df = pd.read_csv(csv_path)
    ids = df['image_id'].tolist()
    X = []  # lista pentru vectorii de caracteristici
    y = []  # lista pentru etichete (daca exista)

    # parcurgem fiecare id si incarcam imaginea corespunzatoare
    for img_id in ids:
        path = os.path.join(img_dir, f"{img_id}.png")       # construim calea catre fisier
        img = Image.open(path).convert('RGB')               # incarcam imaginea si convertim la RGB
        arr = np.asarray(img, dtype=np.float32).ravel()     # transformam in vector 1d float32
        X.append(arr)                                       # adaugam vectorul in lista

    # convertim lista de vectori intr-un array 2d de forma (n_samples, n_features)
    X = np.stack(X)

    if with_labels:
        # extragem etichetele din dataframe
        y = df['label'].values
        return X, y, ids
    else:
        return X, None, ids

# incarcam datele de antrenare si validare
print("loading training and validation data...")
X_train, y_train, train_ids = load_flattened_images(TRAIN_CSV, TRAIN_DIR, with_labels=True)
X_val,   y_val,   val_ids   = load_flattened_images(VAL_CSV,   VAL_DIR,   with_labels=True)
print(f"train samples: {X_train.shape[0]}, validation samples: {X_val.shape[0]}")

# definim pipeline-ul de preprocesare si clasificare
pipeline = Pipeline([
    # pas 1: scalare a caracteristicilor la media 0 si varianta 1
    ('scaler', StandardScaler()),
    # pas 2: reducere dimensionala pastrand 200 componente principale
    ('pca', PCA(n_components=200, random_state=42)),
    # pas 3: clasificator svm cu kernel rbf si parametri impliciti
    ('svc', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
])

# antrenam pipeline-ul pe datele de antrenare
print("training kernel svm pipeline...")
pipeline.fit(X_train, y_train)

# evaluam performanta pe setul de validare
print("evaluating on validation set...")
y_pred_val = pipeline.predict(X_val)
val_acc = accuracy_score(y_val, y_pred_val)
print(f"validation accuracy: {val_acc:.4f}")

# incarcam si prezicem pe datele de test
print("loading and predicting on test data...")
X_test, _, test_ids = load_flattened_images(TEST_CSV, TEST_DIR, with_labels=False)
y_pred_test = pipeline.predict(X_test)

# pregatim fisierul de submissie
submission = pd.DataFrame({
    'image_id': test_ids,
    'label'   : y_pred_test
})
submission.to_csv('kernel_svm_submission.csv', index=False)
print("saved predictions to kernel_svm_submission.csv")
