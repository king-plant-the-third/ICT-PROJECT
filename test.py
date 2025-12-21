import cv2
import numpy as np
import joblib
from PIL import Image
import requests  # Not needed for camera but kept for consistency

# LOAD YOUR SAVED MODEL (must be in same folder)
clf = joblib.load("human_svm_model.joblib")
scaler = joblib.load("scaler.joblib")
pca = joblib.load("pca.joblib")


# Copy your compute_hog function here (from training script)
def compute_hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), nbins=9):
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


# Webcam detection
cap = cv2.VideoCapture("http://192.168.100.5:4444")  # 0 = default camera
img_size = (128, 64)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame (resize + grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size[1], img_size[0]))

    # Extract HOG + predict
    hog = compute_hog(resized)
    hog_scaled = scaler.transform([hog])
    hog_pca = pca.transform(hog_scaled)
    prediction = clf.predict(hog_pca)[0]

    # Show result on frame (BIG TEXT)
    label = "HUMAN DETECTED!" if prediction == 1 else "NO HUMAN"
    color = (0, 255, 0) if prediction == 1 else (0, 0, 255)

    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    cv2.imshow("Human Detector (Press Q to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
