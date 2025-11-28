
"""
Comprehensive script for:
- Train / Evaluate Classification model
- Train YOLOv8 model
- Run YOLO inference
- Run Streamlit app with two tabs (Classification + YOLO detection)

Usage (examples):
python project_scripts.py train_classify
python project_scripts.py train_yolo
streamlit run project_scripts.py -- (this will run the Streamlit app)
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import tempfile

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# For metrics
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ultralytics / YOLO
from ultralytics import YOLO

# Streamlit
import streamlit as st

# ----------------------------
# CONFIG / PATHS — EDIT IF NEEDED
# ----------------------------
ROOT = r"C:\Users\adity\Downloads\Internship\Aerial"
val_name = 'valid' if os.path.exists(os.path.join(ROOT, 'valid')) else 'val' if os.path.exists(os.path.join(ROOT, 'val')) else None
if val_name is None:
    raise FileNotFoundError("No 'valid' or 'val' folder under ROOT. Please create or rename.")

TRAIN_DIR = os.path.join(ROOT, 'train')
VAL_DIR   = os.path.join(ROOT, val_name)
TEST_DIR  = os.path.join(ROOT, 'test')

IMG_SIZE = (224, 224)
BATCH = 32
CLASS_MODE = 'binary'
CLASSIFIER_PATH = os.path.join(os.getcwd(), "bird_drone_classifier.h5")
YOLO_DATA_YAML = os.path.join(os.getcwd(), "data.yaml")
YOLO_BASE = "yolov8n.pt"  # base model (downloaded by ultralytics automatically)
YOLO_SAVE_NAME = "drone_bird_detection"

# ----------------------------
# Utilities
# ----------------------------
def ensure_dirs():
    for p in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing folder: {p}")

def build_classification_model(input_shape=(224,224,3)):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        BatchNormalization(),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        BatchNormalization(),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        BatchNormalization(),

        Conv2D(256, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        BatchNormalization(),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ----------------------------
# Classification training
# ----------------------------
def train_classification(epochs=20):
    ensure_dirs()
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.12,
        height_shift_range=0.12,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=(0.8,1.2)
    )

    valid_gen = ImageDataGenerator(rescale=1./255)
    test_gen  = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode=CLASS_MODE)
    val_data   = valid_gen.flow_from_directory(VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode=CLASS_MODE)
    test_data  = test_gen.flow_from_directory(TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode=CLASS_MODE, shuffle=False)

    model = build_classification_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    callbacks = [
        ModelCheckpoint(CLASSIFIER_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=callbacks)
    model.save(CLASSIFIER_PATH)
    print("Saved classifier to:", CLASSIFIER_PATH)

# ----------------------------
# Classification inference util
# ----------------------------
def classify_image_pil(pil_image, model_path=CLASSIFIER_PATH):
    model = load_model(model_path)
    img = pil_image.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    pred = model.predict(arr)[0][0]
    label = "Drone" if pred > 0.5 else "Bird"
    conf = float(pred if pred > 0.5 else 1 - pred)
    return label, conf

# ----------------------------
# YOLO data.yaml creation
# ----------------------------
def write_yolo_data_yaml():
    # Try to derive class names from classification train folder subfolders
    class_names = []
    for item in sorted(os.listdir(TRAIN_DIR)):
        p = os.path.join(TRAIN_DIR, item)
        if os.path.isdir(p):
            class_names.append(item)
    if len(class_names) == 0:
        # fallback to generic
        class_names = ["bird", "drone"]

    yolo_data = {
        "path": ROOT,
        "train": "train",
        "val": val_name,
        "test": "test",
        "nc": len(class_names),
        "names": class_names
    }

    import yaml
    with open(YOLO_DATA_YAML, "w") as f:
        yaml.dump(yolo_data, f, sort_keys=False)
    print("Wrote YOLO data.yaml:", YOLO_DATA_YAML)
    print(yolo_data)

# ----------------------------
# YOLO training wrapper
# ----------------------------
def train_yolo(epochs=50, imgsz=640, batch=16, base_model=YOLO_BASE, save_name=YOLO_SAVE_NAME):
    write_yolo_data_yaml()
    y = YOLO(base_model)
    y.train(data=YOLO_DATA_YAML, epochs=epochs, imgsz=imgsz, batch=batch, name=save_name)
    print("YOLO training complete. Check runs/detect/" + save_name)

# ----------------------------
# YOLO inference wrapper
# ----------------------------
def yolo_predict_image(pil_image, weights_path=None):
    # save temp image then run
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    pil_image.save(tmp.name)
    model_path = weights_path or "best.pt"
    y = YOLO(model_path)
    results = y(tmp.name)[0]
    annotated = results.plot()  # numpy array
    boxes = []
    if hasattr(results, 'boxes'):
        for b in results.boxes:
            boxes.append({
                "cls": int(b.cls[0]),
                "conf": float(b.conf[0]),
                "xyxy": b.xyxy[0].tolist()
            })
    return annotated, boxes

# ----------------------------
# Streamlit app
# ----------------------------
def run_streamlit_app(yolo_weights="runs/detect/{}/weights/best.pt".format(YOLO_SAVE_NAME), classifier_path=CLASSIFIER_PATH):
    # To run: streamlit run project_scripts.py
    st.set_page_config(page_title="Bird/Drone AI System", layout="wide")
    st.title("🛩️ Bird vs Drone — Classification & YOLO Detection")

    tab1, tab2 = st.tabs(["Classification", "YOLO Detection"])

    # Load models lazily
    classifier = None
    yolo_model = None

    with tab1:
        st.header("Image Classification (Bird vs Drone)")

        uploaded = st.file_uploader("Upload image for classification", type=["jpg","jpeg","png"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded", use_column_width=True)
            label, conf = classify_image_pil(img, model_path=classifier_path)
            st.success(f"Prediction: **{label}** — Confidence: **{conf*100:.2f}%**")

    with tab2:
        st.header("YOLOv8 Object Detection")
        uploaded2 = st.file_uploader("Upload image for detection", type=["jpg","jpeg","png"], key="yolo")
        if uploaded2:
            img = Image.open(uploaded2).convert("RGB")
            st.image(img, caption="Uploaded", use_column_width=True)

            # run YOLO
            weights = yolo_weights if os.path.exists(yolo_weights) else None
            if weights is None:
                st.warning("YOLO weights not found at the default location. Please provide path to a .pt weights file or train YOLO.")
            else:
                annotated, boxes = yolo_predict_image(img, weights_path=weights)
                st.image(annotated, caption="YOLO Annotated", use_column_width=True)
                st.write("Detected objects:")
                for b in boxes:
                    st.write(f"Class {b['cls']} — Conf: {b['conf']:.2f} — Box: {b['xyxy']}")

# ----------------------------
# CLI handler
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", nargs='?', default="run_app", help="train_classify | train_yolo | run_app | check_data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--y_epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    if args.action == "train_classify":
        train_classification(epochs=args.epochs)
    elif args.action == "train_yolo":
        train_yolo(epochs=args.y_epochs, imgsz=args.imgsz)
    elif args.action == "check_data":
        ensure_dirs()
        print("Data folders exist.")
    elif args.action in ("run_app", "streamlit"):
        # Launch streamlit app
        # Note: When running via `streamlit run project_scripts.py` Streamlit will execute this file top-to-bottom.
        # So within Streamlit, call run_streamlit_app() directly below (Streamlit requires __name__ == '__main__').
        run_streamlit_app()
    else:
        print("Unknown action. Choose: train_classify | train_yolo | run_app | check_data")

if __name__ == "__main__":
    # If launched with `streamlit run`, streamlit passes control differently.
    # We guard so Streamlit shows the app:
    if 'streamlit' in sys.argv[0] or os.environ.get('STREAMLIT_RUN'):
        run_streamlit_app()
    else:
        main()
