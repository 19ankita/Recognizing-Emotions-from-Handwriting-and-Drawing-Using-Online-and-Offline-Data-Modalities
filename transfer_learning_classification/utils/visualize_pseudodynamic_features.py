import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from src.pseudo_features import extract_pseudo_dynamic_features


FEATURE_NAMES = [
    "Stroke Length",
    "Mean Thickness",
    "Stroke Segments",
    "Aspect Ratio",
    "Curvature"
]

# --------------------------------------------------
# Utility
# --------------------------------------------------
def load_image(path):

    """
    Load an image from disk and convert it to RGB format.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the input image file.

    Returns
    -------
    numpy.ndarray
        RGB image array of shape (H, W, 3).

    Raises
    ------
    FileNotFoundError
        If the image file cannot be read.
    """

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --------------------------------------------------
# Core function (IMPORTABLE from run_train.py)
# --------------------------------------------------
def visualize_single_image(image_path, save_dir=None):

    """
    Visualize pseudo-dynamic features extracted from a single handwriting image.

    The function loads an image, computes its pseudo-dynamic feature vector,
    and generates a figure showing both the handwriting sample and the
    corresponding feature values.

    Parameters
    ----------
    image_path : str or pathlib.Path
        Path to the handwriting image.
    save_dir : str or pathlib.Path, optional
        Directory in which to save the visualization as a PDF. If None,
        the figure is displayed interactively.

    Returns
    -------
    None
    """

    img = load_image(image_path)
    features = extract_pseudo_dynamic_features(img)

    save_path = None
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "single_image_pseudo_features.pdf"

    plot_single_image(img, features, save_path)


# --------------------------------------------------
# Plot helper
# --------------------------------------------------
def plot_single_image(img, features, save_path=None):

    """
    Plot a handwriting image alongside its pseudo-dynamic feature vector.

    Parameters
    ----------
    img : numpy.ndarray
        RGB handwriting image of shape (H, W, 3).
    features : array-like
        Vector of extracted pseudo-dynamic features.
    save_path : str or pathlib.Path, optional
        File path to save the generated figure. If None, the figure is shown.

    Returns
    -------
    None
    """

    plt.figure(figsize=(10, 4))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Handwriting Sample")
    plt.axis("off")

    # Feature vector
    plt.subplot(1, 2, 2)
    plt.bar(FEATURE_NAMES, features)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Normalized Value")
    plt.title("Pseudo-Dynamic Feature Vector")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()