import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO


def compute_hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), nbins=9):
    """Extract HOG features - like finding human shape edges (ELI5: robot eyes)"""
    img = np.float32(image)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = np.mod(ang, 180.0)

    cell_x, cell_y = pixels_per_cell[1], pixels_per_cell[0]
    sx, sy = img.shape[1] // cell_x, img.shape[0] // cell_y
    bins = np.zeros((sy, sx, nbins), dtype=np.float32)
    bin_width = 180 / nbins

    for i in range(sy):
        for j in range(sx):
            y0, x0 = i * cell_y, j * cell_x
            patch_mag = mag[y0 : y0 + cell_y, x0 : x0 + cell_x].ravel()
            patch_ang = ang[y0 : y0 + cell_y, x0 : x0 + cell_x].ravel()
            for k in range(patch_ang.size):
                angle = patch_ang[k]
                m = patch_mag[k]
                b = int(angle // bin_width) % nbins
                ratio = (angle - (b * bin_width)) / bin_width
                bins[i, j, b] += m * (1 - ratio)
                bins[i, j, (b + 1) % nbins] += m * ratio

    bx = sx - cells_per_block[1] + 1
    by = sy - cells_per_block[0] + 1
    hog_vector = []
    eps = 1e-6
    for i in range(by):
        for j in range(bx):
            block = bins[
                i : i + cells_per_block[0], j : j + cells_per_block[1], :
            ].ravel()
            norm = np.linalg.norm(block)
            block = block / (norm + eps)
            block = np.minimum(block, 0.2)
            block = block / (np.linalg.norm(block) + eps)
            hog_vector.extend(block.tolist())
    return np.array(hog_vector)


def load_dataset(root_folder, img_size=(128, 64)):
    """Load YOUR local folders: human/0 (no humans), human/1 (humans)"""
    X, y = [], []
    categories = ["0", "1"]  # Your folder names

    for label, cat in enumerate(categories):
        folder = os.path.join(root_folder, f"D:/Downloads/human/{cat}")
        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue
        files = (
            glob(os.path.join(folder, "*.png"))
            + glob(os.path.join(folder, "*.jpg"))
            + glob(os.path.join(folder, "*.jpeg"))
        )
        print(f"Found {len(files)} images in {folder}")

        for f in tqdm(files, desc=f"Loading {cat}"):
            img = cv2.imread(f)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (img_size[1], img_size[0]))
            hog = compute_hog(img)
            X.append(hog)
            y.append(label)
    return np.array(X), np.array(y)


# MAIN TRAINING (change 'your_dataset_folder' to your actual path)
dataset_root = "your_dataset_folder"  # e.g., "/home/user/data" or "."
X, y = load_dataset(dataset_root)
print(f"Dataset loaded: X={X.shape}, y={y.shape}")

# Split, scale, PCA, train SVM
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train_s)
X_test_pca = pca.transform(X_test_s)

clf = SVC(kernel="rbf", C=10, gamma=0.001)
clf.fit(X_train_pca, y_train)

# Results
pred = clf.predict(X_test_pca)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))


# PREDICT NEW IMAGE FROM URL
def predict_url(url, model, scaler, pca, img_size=(128, 64)):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("L").resize(img_size)
        img = np.array(img)
        hog = compute_hog(img)
        hog_s = scaler.transform([hog])
        hog_pca = pca.transform(hog_s)
        return model.predict(hog_pca)[0]
    except:
        return None


# Test examples
urls = [
    "https://i.pinimg.com/1200x/ec/e6/7c/ece67cb653a6c07ab10d83e4a95191e8.jpg",  # No human?
    "https://cdn.vntrip.vn/cam-nang/wp-content/uploads/2020/10/bi-quyet-chup-anh-dep-cho-nguoi-khong-an-anh-4.jpg",  # Human?
]
for url in urls:
    pred = predict_url(url, clf, scaler, pca)
    print(f"{url}: {'HUMAN' if pred == 1 else 'NO HUMAN'}")
    import joblib

    # Save ALL needed pieces
joblib.dump(clf, 'human_svm_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(pca, 'pca.joblib')
print("Model saved! Ready for camera detection.")
