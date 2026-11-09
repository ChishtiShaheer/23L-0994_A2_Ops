"""
Tasks 2.2 & 2.3: Baseline CNN and MobileNetV2 Transfer Learning Models
Author: 23L-0994
Project: 23L-0994_A2_Ops
Implements the baseline custom CNN and MobileNetV2 fine-tuning architecture with layer inspections.
"""
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd

def build_baseline_cnn(input_shape=(224, 224, 3), learning_rate=1e-3):
    """
    Task 2.2: Baseline CNN with Convolution, Pooling, Normalization, Dropout, Sigmoid.
    Loss: BinaryCrossentropy
    Optimizer: Adam
    Metrics: Accuracy, Precision, Recall, AUC
    """
    inputs = layers.Input(shape=input_shape, name="input_layer")

    # Rescale pixel values [0, 255] -> [0, 1]
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output_sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Baseline_CNN_23L0994")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )
    return model

def build_transfer_model(input_shape=(224, 224, 3), learning_rate=1e-3):
    """
    Task 2.3: Transfer learning model using MobileNetV2.
    Base model initially frozen with custom classification head.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # MobileNetV2 preprocessing maps [0, 255] to [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # Pretrained base without ImageNet top
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="transfer_gap")(x)
    x = layers.BatchNormalization(name="transfer_bn")(x)
    x = layers.Dense(128, activation="relu", name="transfer_dense")(x)
    x = layers.Dropout(0.4, name="transfer_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="transfer_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Transfer_23L0994")
    model.base_model = base_model  # attach reference for fine-tuning

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )
    return model

def unfreeze_transfer_model(model, unfreeze_layers=20, fine_tune_lr=1e-5):
    """
    Task 2.3: Unfreezes the last block of the base model and recompiles with a low learning rate.
    """
    base_model = model.get_layer("mobilenetv2_1.00_224") if "mobilenetv2_1.00_224" in [l.name for l in model.layers] else model.base_model
    base_model.trainable = True

    # Freeze all layers except the last `unfreeze_layers`
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False

    optimizer = tf.keras.optimizers.Adam(learning_rate=fine_tune_lr)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    trainable_layers = [layer.name for layer in base_model.layers if layer.trainable]
    print(f"[+] Base model unfrozen: {len(trainable_layers)} trainable layers in top block.")
    return trainable_layers

def export_model_summary_table(model, output_csv="reports/baseline_cnn_layers.csv"):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    layer_data = []
    for layer in model.layers:
        layer_data.append({
            "Layer Name": layer.name,
            "Type": layer.__class__.__name__,
            "Output Shape": str(layer.output_shape),
            "Param Count": layer.count_params(),
            "Trainable": layer.trainable
        })
    df = pd.DataFrame(layer_data)
    df.to_csv(output_csv, index=False)
    print(f"[+] Model architecture summary exported to {output_csv}")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    print("--- BASELINE CNN SUMMARY (Task 2.2) ---")
    cnn = build_baseline_cnn()
    cnn.summary()
    export_model_summary_table(cnn)

    print("\n--- MOBILENETV2 TRANSFER SUMMARY (Task 2.3) ---")
    mobilenet = build_transfer_model()
    mobilenet.summary()
