import json
import os
import pickle

import numpy as np

from src import config


def _class_from_filename(path):
    """'data\\Aamir_Khan\\Aamir.100.jpg' -> 'Aamir_Khan' (handles / and \\)."""
    norm = str(path).replace("\\", "/")
    parts = norm.split("/")
    # class folder is the component right after the 'data' directory
    if "data" in parts:
        idx = parts.index("data")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    # fallback: second-to-last component (…/<class>/<file>)
    return parts[-2]


def build_label_map():
    
    filenames = pickle.load(open(config.FILENAMES_PKL, "rb"))
    classes = sorted({_class_from_filename(f) for f in filenames})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return classes, class_to_idx


def load_embedding_dataset():
    
    embeddings = pickle.load(open(config.EMBEDDING_PKL, "rb"))
    filenames = pickle.load(open(config.FILENAMES_PKL, "rb"))

    classes, class_to_idx = build_label_map()
    X = np.asarray(embeddings, dtype="float32")
    y = np.asarray([class_to_idx[_class_from_filename(f)] for f in filenames], dtype="int64")
    return X, y, classes


def save_labels(classes, path=None):
   
    path = path or config.LABELS_PATH
    config.ensure_models_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(i): c for i, c in enumerate(classes)}, f, indent=2)
    return path


def load_labels(path=None):
    path = path or config.LABELS_PATH
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # return list where index == class id
    return [raw[str(i)] for i in range(len(raw))]


def build_image_generators(img_size=None, batch_size=32, validation_split=0.2,
                           preprocessing_function=None):
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    img_size = img_size or config.IMG_SIZE

    common = dict(validation_split=validation_split)
    if preprocessing_function is not None:
        common["preprocessing_function"] = preprocessing_function
    else:
        common["rescale"] = 1.0 / 255

    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        **common,
    )
    val_datagen = ImageDataGenerator(**common)

    train_gen = train_datagen.flow_from_directory(
        config.DATA_DIR,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        config.DATA_DIR,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )
    classes = [c for c, _ in sorted(train_gen.class_indices.items(), key=lambda kv: kv[1])]
    return train_gen, val_gen, classes
