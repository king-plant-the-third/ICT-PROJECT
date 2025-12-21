import os
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score
from scipy import ndimage


# Extract features from one image
def get_features(img_path):
    img = Image.open(img_path).convert("L").resize((64, 64))
    img_array = np.array(img)

    # Feature 1: Edge count
    edges = img.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges)
    edge_count = np.sum(edge_array > 50)

    # Feature 2: Object count using connected components
    binary = img_array > img_array.mean()
    labeled_array, object_count = ndimage.label(binary)

    # Feature 3: Standard deviation (texture/contrast)
    std_dev = img_array.std()

    # Feature 4: Variance (how spread out pixel values are)
    variance = img_array.var()

    # Feature 5: Edge density (edges relative to image size)
    edge_density = edge_count / (64 * 64)

    return edge_count, object_count, std_dev, variance, edge_density


# Load all images
def load_data(human_folder, non_human_folder):
    features = []
    labels = []

    # Load human images (label = 1)
    for file in os.listdir(human_folder):
        try:
            path = os.path.join(human_folder, file)
            feats = get_features(path)
            features.append(feats)
            labels.append(1)
        except:
            pass

    # Load non-human images (label = 0)
    for file in os.listdir(non_human_folder):
        try:
            path = os.path.join(non_human_folder, file)
            feats = get_features(path)
            features.append(feats)
            labels.append(0)
        except:
            pass

    return np.array(features), np.array(labels)


#dataset paths
human_path = "E:/PycharmProjects/dataset/human"
non_human_path = "E:/PycharmProjects/dataset/non_human"

# Loading data
X, y = load_data(human_path, non_human_path)

print(f"Total images loaded: {len(X)}")
print(f"Humans: {sum(y)}, Non-humans: {len(y) - sum(y)}")

# Split into train and test (20% test, 80% train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add polynomial features (degree 2 = includes x^2, xy, etc.)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

print(f"\nOriginal features: {X_train_scaled.shape[1]}")
print(f"After polynomial expansion: {X_train_poly.shape[1]}")

# Train model with polynomial features
model = LogisticRegression(max_iter=2000, class_weight='balanced', C=0.1)
model.fit(X_train_poly, y_train)

# Predict
y_pred_class = model.predict(X_test_poly)
y_pred_proba = model.predict_proba(X_test_poly)[:, 1]

# Calculate statistics
accuracy = accuracy_score(y_test, y_pred_class)
correct = sum(y_test == y_pred_class)
total_test = len(y_test)

actual_humans = sum(y_test == 1)
actual_non_humans = sum(y_test == 0)

correct_humans = sum((y_test == 1) & (y_pred_class == 1))
correct_non_humans = sum((y_test == 0) & (y_pred_class == 0))

print(f"\n=== TEST SET RESULTS (POLYNOMIAL FEATURES) ===")
print(f"Total test images: {total_test}")
print(f"Actual Humans: {actual_humans}")
print(f"Actual Non-Humans: {actual_non_humans}")
print(f"\nPredicted Humans: {sum(y_pred_class == 1)}")
print(f"Predicted Non-Humans: {sum(y_pred_class == 0)}")
print(f"\nCorrectly predicted Humans: {correct_humans}/{actual_humans} ({100 * correct_humans / actual_humans:.1f}%)")
print(f"Correctly predicted Non-Humans: {correct_non_humans}/{actual_non_humans} ({100 * correct_non_humans / actual_non_humans:.1f}%)")
print(f"\nTotal Correct: {correct}/{total_test}")
print(f"Accuracy: {accuracy * 100:.2f}%")

# ============ WINDOW 1: Predictions ============
plt.figure(figsize=(10, 7))
correct_mask = y_test == y_pred_class

plt.scatter(X_test[correct_mask & (y_test == 1), 0],
            X_test[correct_mask & (y_test == 1), 1],
            c='green', marker='o', label='Correct Human', alpha=0.7, s=100)
plt.scatter(X_test[correct_mask & (y_test == 0), 0],
            X_test[correct_mask & (y_test == 0), 1],
            c='blue', marker='o', label='Correct Non-Human', alpha=0.7, s=100)
plt.scatter(X_test[~correct_mask & (y_test == 1), 0],
            X_test[~correct_mask & (y_test == 1), 1],
            c='red', marker='x', label='Wrong (Was Human)', alpha=0.9, s=150, linewidths=3)
plt.scatter(X_test[~correct_mask & (y_test == 0), 0],
            X_test[~correct_mask & (y_test == 0), 1],
            c='orange', marker='x', label='Wrong (Was Non-Human)', alpha=0.9, s=150, linewidths=3)
plt.xlabel('Edge Count', fontsize=12)
plt.ylabel('Object Count', fontsize=12)
plt.title(f'Predictions | Accuracy: {accuracy * 100:.1f}%', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# ============ WINDOW 2: Prediction Confidence ============
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_proba, cmap='RdYlGn',
                      s=100, alpha=0.7, edgecolors='black', vmin=0, vmax=1)
plt.colorbar(scatter, label='Probability of Human')
plt.xlabel('Edge Count', fontsize=12)
plt.ylabel('Object Count', fontsize=12)
plt.title('Prediction Confidence', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# ============ WINDOW 3: Regression Curve ============
plt.figure(figsize=(10, 7))

# Sort test data by edge count for a smooth line
sort_idx = np.argsort(X_test[:, 0])
X_test_sorted = X_test[sort_idx]
y_pred_proba_sorted = y_pred_proba[sort_idx]
y_test_sorted = y_test.values[sort_idx] if hasattr(y_test, 'values') else y_test[sort_idx]

# Plot the regression line
plt.plot(X_test_sorted[:, 0], y_pred_proba_sorted, 'b-', linewidth=2, label='Prediction Curve')

# Plot actual values as dots
plt.scatter(X_test[y_test == 1, 0], y_test[y_test == 1],
            c='red', marker='o', s=80, alpha=0.6, label='Actual Human', edgecolors='black')
plt.scatter(X_test[y_test == 0, 0], y_test[y_test == 0],
            c='blue', marker='o', s=80, alpha=0.6, label='Actual Non-Human', edgecolors='black')

# Add threshold line
plt.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')

plt.xlabel('Edge Count', fontsize=12)
plt.ylabel('Prediction Value', fontsize=12)
plt.title('Regression Curve', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 1.1)
plt.tight_layout()

plt.show()
