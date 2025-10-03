import yaml
import mlflow
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
    ReLU,
    Input,
)
from tensorflow.keras.applications import ResNet50


def load_config(config_path: str) -> dict:
    """
    Loads training configs from YAML file

    Args:
        config_path

    Returns:
        config dict that contains training parameters
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def build_cnn_model(input_shape: tuple[int, int, int] = (224, 224, 3), num_classes=43):
    """
    Builds a simple CNN model with Conv2D, MaxPool, Dense, Droput & BatchNormalization

    Args:
        input_shape:
        num_classes:
    Returns:
        Compiled CNN model
    """
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(256, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Conv2D(512, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(
                num_classes, activation="softmax"
            ),  # 43 because it the number of classes for current project
        ]
    )
    return model


def build_transfer_model(
    input_shape: tuple[int, int, int] = (224, 224, 3), num_classes: int = 43
):
    """
    Loads a pretrained model of ResNet50 (imagenet weights, include_top=False)

    Args:
        input_shape:
        num_classes:
    Returns:
        Compiled transfer model using ResNet50
    """
    base_model = ResNet50(
        weights="imagenet", include_top=False, input_shape=input_shape
    )

    base_model.trainable = True
    classifier = Sequential(
        [
            GlobalAveragePooling2D(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )

    output = classifier(base_model.output)
    model = Model(inputs=base_model.input, outputs=output)
    return model


def train_and_evaluate(model, train_data, val_data, test_data, config: dict):
    """
    Args:
        model:
        train_data:
        val_data:
        test_data:
        config:
    Returns:
        History and the evaluated metrics
    """
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    epochs = config["epochs"]
    batch_size = config["batch_size"]

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
    )
    evaluate_metrics = model.evaluate(X_test, y_test)

    #save_dir = "../models"
    #os.makedirs(save_dir, exist_ok=True)
    model_name = config.get("model_name", "model")
    model.save(f"../models/{model_name}.h5")
    return history, evaluate_metrics


def log_experiment_mflow(model, history, metrics: list[str], config: dict):
    """
    Args:
        model:
        history:
        metrics:
        config:
    Returns:
        Compiled CNN model
    """
    metrics_list = ["val_" + item for item in metrics]
    epochs = config["epochs"]
    batch_size = config["batch_size"]

    with mlflow.start_run():
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        # mlflow.log_param("learning_rate", 0.0001)
        mlflow.log_param("model_type", model)

        for item in metrics_list:
            mlflow.log_metric(item, max(history.history[item]))

        mlflow.log_artifact("reports/confusion_matrix.png")
        mlflow.keras.log_model(model, "cnn_gtsrb_model")
