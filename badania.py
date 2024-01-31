import os
import cv2
import numpy as np
from skimage import feature
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from random_clasifier import RandomClassfier
from sklearn.neighbors import KNeighborsClassifier

def calculate_lbp_histogram(image):
    lbp = feature.local_binary_pattern(image, P=16, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(0, 256))
    return hist

def create_dataset(folder_path, activity_sizes, target_size=(200, 200)):
    data = []
    dataGS = []
    target = []
    clazz = 1
    num_of_photo = 1

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            print(filename)
            activity_number = clazz
            image_path = os.path.join(folder_path, filename)
            
            img = cv2.imread(image_path)

            if img is not None:
                img_GS = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            
                lbp_features = calculate_lbp_histogram(img_GS)
                img_resized = cv2.resize(img, target_size)
                
                flattened_pixels = img_resized.flatten()
                dataGS.append(lbp_features)
                data.append(flattened_pixels)
                target.append(activity_number)
                
                num_of_photo = num_of_photo + 1
                if activity_sizes[clazz - 1] == num_of_photo - 1:
                    num_of_photo = 1
                    clazz = clazz + 1

    return np.vstack(data), np.vstack(dataGS), np.array(target)

def apply_pca(X, n_components=50):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca

folder_path = 'dataset/output'
activity_sizes = [284, 259, 200, 212, 295, 288, 203, 189, 256, 287, 273, 228, 251, 199, 292, 295, 191, 203, 289, 260, 200, 235, 245, 259, 293, 296, 185, 251, 214, 241, 197, 193, 202, 230, 293, 182, 223, 210, 183, 246]

data, dataGS, target = create_dataset(folder_path, activity_sizes)


n_components = 50  # Możesz dostosować tę wartość


scaler = StandardScaler()
data_pca = apply_pca(data, n_components)
data_normalized = scaler.fit_transform(data)
data_normalizedGS = scaler.fit_transform(dataGS)
data_pca_normalized = scaler.fit_transform(data_pca)
data_normalized_pca = apply_pca(data_normalized, n_components)
datas = [ dataGS, data_normalizedGS]

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(data_pca, target, test_size=0.2, random_state=42)
X_train3, X_test3, y_train3, y_test3 = train_test_split(data_normalized, target, test_size=0.2, random_state=42)
X_train4, X_test4, y_train4, y_test4 = train_test_split(data_pca_normalized, target, test_size=0.2, random_state=42)
X_train5, X_test5, y_train5, y_test5 = train_test_split(data_normalized_pca, target, test_size=0.2, random_state=42)
X_train6, X_test6, y_train6, y_test6 = train_test_split(dataGS, target, test_size=0.2, random_state=42)
X_train7, X_test7, y_train7, y_test7 = train_test_split(data_normalizedGS, target, test_size=0.2, random_state=42)
all_data = []

all_data.append((X_train, X_test, y_train, y_test))
all_data.append((X_train2, X_test2, y_train2, y_test2))
all_data.append((X_train3, X_test3, y_train3, y_test3))
all_data.append((X_train4, X_test4, y_train4, y_test4))
all_data.append((X_train5, X_test5, y_train5, y_test5))
all_data.append((X_train6, X_test6, y_train6, y_test6))
all_data.append((X_train7, X_test7, y_train7, y_test7))
clfs = [
    RandomForestClassifier(n_estimators=100, random_state=42),
    RandomClassfier(random_state=42),
    KNeighborsClassifier(n_neighbors=40, random_state = 42),
    KNeighborsClassifier(n_neighbors=3, random_state = 42)
]
accuracy = []
for i, clf in enumerate(clfs):
    tqdm.write(f'Trenowanie klasyfikatora {i + 1}...')
    
    for j, data_set in enumerate(all_data):
        X_train, X_test, y_train, y_test = data_set
        
        with tqdm(total=len(X_train), desc="Training", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            clf.fit(X_train, y_train)
            pbar.update(len(X_train))
        
        with tqdm(total=len(X_test), desc="Predicting", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            y_pred = clf.predict(X_test)
            pbar.update(len(X_test))
        
        acc = accuracy_score(y_test, y_pred)
        tqdm.write(f'Accuracy dla klasyfikatora {i + 1} i zestawu danych {j + 1}: {acc}')
        accuracy.append(acc)

print("\\begin{tabular}{|c|c|c|c|c|}")
print("\\hline")
print(" & RFC & RC 2 & KNN k=3 & KNN k=40 \\\\")
print("\\hline")

for i in range(len(all_data)):
    clf_name = f'data {i+1}'
    acc_row = ' & '.join([f'{acc:.4f}' for acc in accuracy[i::len(all_data)]])
    print(f"{clf_name} & {acc_row} \\\\")
    print("\\hline")

print("\\end{tabular}")