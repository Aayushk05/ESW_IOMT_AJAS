# -------------------- Installs (uncomment if first run) --------------------
# !pip install wfdb tensorflow scikit-learn matplotlib pandas -q

import os
import random
import numpy as np
import wfdb
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    ConfusionMatrixDisplay, classification_report, average_precision_score
)
import pandas as pd

# -------------------- User-provided paths --------------------
LOCAL_DATA_DIR = 'physionet_data'
GDRIVE_DATA_PATH = '/content/drive/My Drive/Colab_Datasets/physionet_data'

# -------------------- Mount Drive (Colab) --------------------
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
except Exception:
    pass

if os.path.exists(GDRIVE_DATA_PATH):
    DATA_DIR = GDRIVE_DATA_PATH
    print("Using dataset from Google Drive:", DATA_DIR)
elif os.path.exists(LOCAL_DATA_DIR):
    DATA_DIR = LOCAL_DATA_DIR
    print("Using dataset from local path:", DATA_DIR)
else:
    raise FileNotFoundError(f"Dataset not found. Checked:\n - {GDRIVE_DATA_PATH}\n - {LOCAL_DATA_DIR}\n")

# -------------------- Reproducibility --------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------- Hyperparams --------------------
ANNOTATION_EXTENSION = 'apn'
EPOCH_SECONDS = 30.0
OVERLAP = 0.5
LOWCUT = 0.5
HIGHCUT = 40.0
FILTER_ORDER = 4
BATCH_SIZE = 64
EPOCHS = 20  # ✅ slightly higher, early stopping will handle overfitting
MODEL_SAVE_PATH = 'best_apnea_model.keras'

# -------------------- Utility Functions --------------------
def safe_decode(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else x

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    data = np.asarray(data, dtype=np.float64)
    if np.isnan(data).any():
        nans = np.isnan(data)
        if nans.all():
            return np.zeros_like(data)
        not_nans = ~nans
        data[nans] = np.interp(np.nonzero(nans)[0], np.nonzero(not_nans)[0], data[not_nans])
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    padlen = 3 * max(len(a), len(b))
    if len(data) <= padlen:
        return np.convolve(data, b, mode='same')
    return filtfilt(b, a, data)

# -------------------- Record discovery --------------------
def discover_records(local_dir):
    basenames = [f[:-4] for f in os.listdir(local_dir) if f.lower().endswith('.hea')]
    basenames = sorted(set(basenames))
    print(f"Discovered {len(basenames)} records in {local_dir}")
    return basenames

# -------------------- Windowing & labeling --------------------
def build_windows_from_record(record_name, local_dir, epoch_seconds=30.0, overlap=0.5,
                              lowcut=0.5, highcut=40.0, filter_order=4, ann_ext='apn'):
    record_path = os.path.join(local_dir, record_name)
    try:
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, ann_ext)
    except Exception as e:
        print(f"  - Warning: Failed to read {record_name}: {e}")
        return [], [], []

    fs = int(record.fs)
    signal = record.p_signal[:, 0]

    try:
        filtered = bandpass_filter(signal, lowcut, highcut, fs, order=filter_order)
    except Exception as e:
        print(f"  - Warning: Filtering failed for {record_name}: {e}")
        filtered = signal

    label_signal = np.zeros(len(filtered), dtype=np.uint8)
    for sidx, sym in zip(annotation.sample, annotation.symbol):
        sym = safe_decode(sym)
        if sym == 'A':
            dur_samples = int(epoch_seconds * fs)
            end = min(len(label_signal), sidx + dur_samples)
            label_signal[sidx:end] = 1

    window_size = int(epoch_seconds * fs)
    step = max(1, int(window_size * (1 - overlap)))

    windows, labels, groups = [], [], []
    for start in range(0, len(filtered) - window_size + 1, step):
        w = filtered[start:start + window_size]
        lbl = 1 if label_signal[start:start + window_size].mean() > 0.5 else 0
        windows.append(w)
        labels.append(lbl)
        groups.append(record_name)
    return windows, labels, groups

def build_dataset(local_dir, record_names=None, **window_kwargs):
    if record_names is None:
        record_names = discover_records(local_dir)
    all_wins, all_lbls, all_groups = [], [], []
    for rec in record_names:
        wins, lbls, groups = build_windows_from_record(rec, local_dir, **window_kwargs)
        all_wins.extend(wins)
        all_lbls.extend(lbls)
        all_groups.extend(groups)
    X = np.stack(all_wins, axis=0).astype(np.float32)
    y = np.array(all_lbls, dtype=np.int32)
    groups = np.array(all_groups)
    return X, y, groups

# -------------------- Model Definition --------------------
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, Add,
    MaxPooling1D, GlobalAveragePooling1D, Dropout, Dense
)
from tensorflow.keras.models import Model

def resnet_block(input_tensor, filters, kernel_size=7):
    x = Conv1D(filters, kernel_size, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    shortcut = Conv1D(filters, 1, padding='same')(input_tensor)
    shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 7, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = resnet_block(x, 64, kernel_size=7)
    x = MaxPooling1D(2)(x)
    x = resnet_block(x, 128, kernel_size=5)
    x = MaxPooling1D(2)(x)
    x = resnet_block(x, 256, kernel_size=3)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name='auc')])
    return model

# -------------------- Main Pipeline --------------------
print("Building dataset from:", DATA_DIR)
X, y, groups = build_dataset(DATA_DIR,
                             epoch_seconds=EPOCH_SECONDS,
                             overlap=OVERLAP,
                             lowcut=LOWCUT,
                             highcut=HIGHCUT,
                             filter_order=FILTER_ORDER,
                             ann_ext=ANNOTATION_EXTENSION)
print(f'Extracted windows: {X.shape}, labels distribution: {np.bincount(y)}')
X = X[:, :, None]

# -------------------- Group-aware splits --------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train_full, X_test = X[train_idx], X[test_idx]
y_train_full, y_test = y[train_idx], y[test_idx]
groups_train_full, groups_test = groups[train_idx], groups[test_idx]

# Validation split (group-aware)
gss_val = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
train_idx, val_idx = next(gss_val.split(X_train_full, y_train_full, groups_train_full))
X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

print(f'Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')

# -------------------- Standardization --------------------
scaler = StandardScaler()
X_train_flat = X_train.reshape(X_train.shape[0], -1)
scaler.fit(X_train_flat)
X_train = scaler.transform(X_train_flat).reshape(X_train.shape)
X_val = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape)
X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)

# -------------------- Class weights --------------------
cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {int(k): float(v) for k, v in zip(np.unique(y_train), cw)}
print('Class weights:', class_weight_dict)

# -------------------- Training --------------------
model = build_model(input_shape=(X_train.shape[1], 1))
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=8, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_auc', mode='max', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=4)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# -------------------- Evaluation --------------------
print('\nEvaluating on test set...')
y_pred_proba = model.predict(X_test).ravel()
y_pred = (y_pred_proba > 0.5).astype(int)
test_auc = roc_auc_score(y_test, y_pred_proba)
print(f'Test ROC AUC: {test_auc:.4f}')
print(classification_report(y_test, y_pred, target_names=['Normal', 'Apnea']))

# -------------------- Threshold Optimization & Plots --------------------
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print("\n🔧 Optimal Threshold (Max F1): {:.3f}".format(best_thresh))
print("Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(
    precision[best_idx], recall[best_idx], best_f1))

target_recall = 0.90
recall_idx = np.argmin(np.abs(recall - target_recall))
recall_thresh = thresholds[recall_idx]
print("\n🎯 Threshold for Recall ≈ 0.90: {:.3f}".format(recall_thresh))
print("Precision: {:.3f}, Recall: {:.3f}".format(
    precision[recall_idx], recall[recall_idx]))

y_pred_optimal = (y_pred_proba >= best_thresh).astype(int)
print("\n🧾 Classification Report (Optimal Threshold):")
print(classification_report(y_test, y_pred_optimal, target_names=['Normal', 'Apnea']))

plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision[:-1], label='Precision', linewidth=2)
plt.plot(thresholds, recall[:-1], label='Recall', linewidth=2)
plt.plot(thresholds, f1_scores[:-1], label='F1 Score', linewidth=2)
plt.axvline(best_thresh, color='r', linestyle='--', label=f'Best F1 ({best_thresh:.3f})')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision, Recall, and F1 vs. Decision Threshold')
plt.legend()
plt.grid(True)
plt.show()

metrics_df = pd.DataFrame({
    'threshold': thresholds,
    'precision': precision[:-1],
    'recall': recall[:-1],
    'f1_score': f1_scores[:-1]
})
metrics_df.to_csv('threshold_metrics.csv', index=False)
print("\n📁 Saved threshold metrics to 'threshold_metrics.csv'")

# -------------------- ROC / PR / Confusion Matrix --------------------
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
ap = average_precision_score(y_test, y_pred_proba)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'AUC = {test_auc:.3f}')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(recall, precision, label=f'AP = {ap:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision–Recall Curve')
plt.legend()
plt.show()

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Apnea']).plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

print(f'✅ Best model saved to {MODEL_SAVE_PATH}')
print('Done.')