# hand_gesture_recognition_no_tf.py

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC

# -------------------------------
# 1. Load Dataset
# -------------------------------

dataset_path = r"C:\AC DSA\aProdigyProjectML\leapGestRecog\leapGestRecog"  # ✅ Correct this path as needed

X = []
y = []

# Check dataset path
if not os.path.exists(dataset_path):
    print(f"❌ Dataset path does not exist: {dataset_path}")
    exit()

# ✅ Only keep subject folders like '00' to '09'
subjects = [s for s in os.listdir(dataset_path) if s.isdigit() and len(s) == 2]
print("📂 Subjects found:", subjects)

# Load images
for subject in subjects:
    subject_path = os.path.join(dataset_path, subject)
    if os.path.isdir(subject_path):
        for gesture in os.listdir(subject_path):
            gesture_path = os.path.join(subject_path, gesture)
            if os.path.isdir(gesture_path):
                for img_file in os.listdir(gesture_path):
                    if img_file.endswith('.png'):
                        img_path = os.path.join(gesture_path, img_file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, (64, 64))
                            X.append(img.flatten())  # Flatten to 4096 features
                            y.append(gesture)

X = np.array(X)
y = np.array(y)

print("✅ Dataset Loaded:", X.shape, y.shape)
print("🏷️ Classes:", np.unique(y))

if len(X) == 0 or len(y) == 0:
    print("❌ No data found. Check the dataset structure and path.")
    exit()

# -------------------------------
# 2. Encode Labels
# -------------------------------

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -------------------------------
# 3. Train/Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# -------------------------------
# 4. Train SVM Model
# -------------------------------

print("\n🚀 Training SVM model...")
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# -------------------------------
# 5. Evaluate Model
# -------------------------------

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Test Accuracy: {acc * 100:.2f}%")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# -------------------------------
# 6. Show a Single Prediction
# -------------------------------

def show_prediction(index):
    img = X_test[index].reshape(64, 64)
    true_label = le.inverse_transform([y_test[index]])[0]
    pred_label = le.inverse_transform([y_pred[index]])[0]

    plt.imshow(img, cmap='gray')
    plt.title(f'True: {true_label} | Pred: {pred_label}')
    plt.axis('off')
    plt.show()

show_prediction(0)

# -------------------------------
# 7. Show 5 Random Predictions
# -------------------------------

sample_indices = random.sample(range(len(X_test)), 5)
fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for ax, i in zip(axes, sample_indices):
    img = X_test[i].reshape(64, 64)
    true_label = le.inverse_transform([y_test[i]])[0]
    pred_label = le.inverse_transform([clf.predict([X_test[i]])[0]])[0]

    ax.imshow(img, cmap='gray')
    ax.set_title(f'True: {true_label}\nPred: {pred_label}', fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.show()
