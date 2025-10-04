import numpy as np
from PIL import Image
import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_preprocess_image(img_path: str, target_size=(224, 224)) -> np.ndarray:
    """
    Loads and preprocess a single image

    Args:
        img_path: Path to the image file
        target_size: Target size for resizing (height, width)

    Returns:
        Preprocessed image as numpy array
    """
    image = Image.open(img_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    image = np.array(image)
    return image


def augment_image(image) -> np.ndarray:
    """
    Apply data augmentation to a singel image.

    Arg:
        image: Input image as numpy array
    Return:
        Augmented image as numpy array
    """
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=False,
    )

    image = np.expand_dims(image, axis=0)
    augmented = next(datagen.flow(image, batch_size=1))[0]
    return augmented


def prepare_datasets(
    train_dir: str, test_dir: str, val_size: float = 0.2
) -> tuple[DirectoryIterator, DirectoryIterator, DirectoryIterator, list[str]]:
    """
    Prepare train, validation, and test datasets for GTSRB.

    Args:
        train_dir: Path to the train directory
        test_dir: Path to the test directory
        val_size: the size of the validation from the training dataset
    Returns:
        train_gen, val_gen, test_gen, class_names
    """

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=val_size,
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        subset="training",
    )

    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        subset="validation",
    )

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
    )

    class_names = list(train_gen.class_indices.keys())

    return train_gen, val_gen, test_gen, class_names
