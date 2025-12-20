import os
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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


# Your dataset paths
human_path = "E:/PycharmProjects/dataset/human"
non_human_path = "E:/PycharmProjects/dataset/non_human"

# Load data
X, y = load_data(human_path, non_human_path)

print(f"Total images loaded: {len(X)}")
print(f"Humans: {sum(y)}, Non-humans: {len(y) - sum(y)}")

# Split into train and test (30% test, 70% train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Use LOGISTIC REGRESSION instead of Linear Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred_class = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of being human

# Calculate statistics
accuracy = accuracy_score(y_test, y_pred_class)
correct = sum(y_test == y_pred_class)
total_test = len(y_test)

actual_humans = sum(y_test == 1)
actual_non_humans = sum(y_test == 0)

correct_humans = sum((y_test == 1) & (y_pred_class == 1))
correct_non_humans = sum((y_test == 0) & (y_pred_class == 0))

print(f"\n=== TEST SET RESULTS ===")
print(f"Total test images: {total_test}")
print(f"Actual Humans: {actual_humans}")
print(f"Actual Non-Humans: {actual_non_humans}")
print(f"\nPredicted Humans: {sum(y_pred_class == 1)}")
print(f"Predicted Non-Humans: {sum(y_pred_class == 0)}")
print(f"\nCorrectly predicted Humans: {correct_humans}/{actual_humans}")
print(f"Correctly predicted Non-Humans: {correct_non_humans}/{actual_non_humans}")
print(f"\nTotal Correct: {correct}/{total_test}")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print feature importance
print(f"\n=== FEATURE COEFFICIENTS ===")
feature_names = ['Edge Count', 'Object Count', 'Std Dev', 'Variance', 'Edge Density']
for i, name in enumerate(feature_names):
    print(f"{name}: {model.coef_[0][i]:.6f}")
print(f"Intercept: {model.intercept_[0]:.6f}")

# Plot
plt.figure(figsize=(14, 6))

# Plot 1: Predictions
plt.subplot(1, 2, 1)
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
plt.xlabel('Edge Count')
plt.ylabel('Object Count')
plt.title('Predictions')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Prediction confidence
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_proba, cmap='RdYlGn',
                      s=100, alpha=0.7, edgecolors='black', vmin=0, vmax=1)
plt.colorbar(scatter, label='Probability of Human')
plt.xlabel('Edge Count')
plt.ylabel('Object Count')
plt.title('Prediction Confidence')
plt.grid(True, alpha=0.3)

plt.suptitle(f'Logistic Regression | Accuracy: {accuracy * 100:.1f}%', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
