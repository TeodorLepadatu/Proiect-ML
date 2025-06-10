import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from PIL import Image

# ----------------------------------------
# configurare cai si hiperparametri
# ----------------------------------------
train_csv      = 'train.csv'            # fisier csv cu id-urile de antrenare
val_csv        = 'validation.csv'       # fisier csv cu id-urile de validare
test_csv       = 'test.csv'             # fisier csv cu id-urile de test
train_dir      = 'train'                # director cu imaginile de antrenare
val_dir        = 'validation'           # director cu imaginile de validare
test_dir       = 'test'                 # director cu imaginile de test
submission_csv = 'knn_submission.csv'   # fisier pentru rezultatele de test
img_size       = (100, 100)             # dimensiunea la care redimensionam imaginile
k              = 2                      # numarul de vecini in KNN
apply_scale    = True                   # flag pentru scalare
pca_components = 200                    # numar de componente PCA (None pentru fara PCA)

def load_dataset(csv_path, img_dir, with_labels=True):
    """
    incarca imaginile flatten-ate si, optional, etichetele
    """
    df     = pd.read_csv(csv_path)           # citim csv-ul
    ids    = df['image_id'].tolist()         # lista de id-uri
    images = []                              # lista de vectori imagine
    labels = []                              # lista de etichete

    # iteram prin fiecare id si incarcam imaginea
    for image_id in tqdm(ids, desc=f'loading {img_dir}'):
        path = os.path.join(img_dir, f"{image_id}.png")    # construim calea
        img  = Image.open(path).convert('RGB')              # incarcam si convertim la RGB
        # redimensionam, normalizam in [0,1] si flatten
        arr = np.asarray(img.resize(img_size), dtype=np.float32).ravel() / 255.0
        images.append(arr)
        if with_labels:
            # extragem eticheta corespunzatoare din DataFrame
            labels.append(df.loc[df['image_id']==image_id, 'label'].values[0])

    X = np.stack(images)                        # array shape=(n_samples, n_features)
    y = np.array(labels) if with_labels else None
    return X, y, ids

# incarcam datele de antrenare, validare si test
X_train, y_train, _ = load_dataset(train_csv, train_dir, with_labels=True)
X_val,   y_val,   _ = load_dataset(val_csv,   val_dir,   with_labels=True)
X_test,  _,     test_ids = load_dataset(test_csv,  test_dir,  with_labels=False)

# ----------------------------------------
# preprocesare: scalare si PCA
# ----------------------------------------
if apply_scale:
    scaler  = StandardScaler()               # initializam scaler
    X_train = scaler.fit_transform(X_train)   # fit pe datele de antrenare
    X_val   = scaler.transform(X_val)         # transform pe datele de validare
    X_test  = scaler.transform(X_test)        # transform pe datele de test

if pca_components is not None:
    pca     = PCA(n_components=pca_components, random_state=42)
    X_train = pca.fit_transform(X_train)      # fit + transform pe antrenare
    X_val   = pca.transform(X_val)            # transform pe validare
    X_test  = pca.transform(X_test)           # transform pe test

# ----------------------------------------
# antrenare model KNN
# ----------------------------------------
knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)  # initializam KNN
knn.fit(X_train, y_train)                             # antrenam KNN

# ----------------------------------------
# evaluare pe setul de validare
# ----------------------------------------
y_pred_val = knn.predict(X_val)                       # predictii validare
val_acc     = accuracy_score(y_val, y_pred_val)       # calcul acuratete
print(f'validation accuracy (k={k}): {val_acc:.4f}')

# ----------------------------------------
# predictii pe setul de test si salvare submissie
# ----------------------------------------
y_pred_test  = knn.predict(X_test)                    # predictii test
submission   = pd.DataFrame({'image_id': test_ids, 'label': y_pred_test})
submission.to_csv(submission_csv, index=False)        # salvam csv-ul de submissie
print(f'submission file saved to {submission_csv}')
