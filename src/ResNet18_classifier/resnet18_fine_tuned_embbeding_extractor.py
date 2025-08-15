"""
This script extracts deep learning features (embeddings) from a fine-tuned ResNet-18 model.

The script is structured into three main parts:
1.  A `ResNet18FeatureExtractor` class that loads a pre-trained PyTorch ResNet-18 model,
    configures its final layer to an identity function to output the 512-dimensional
    embedding vector, and freezes the model's weights.
2.  A set of image preprocessing functions that resize and convert images to a PyTorch
    tensor format, which is required by the model.
3.  An `extract_embeddings` function that iterates through a list of image paths from a
    CSV file, runs each image through the feature extractor, and saves the resulting
    embeddings to a new CSV file.

The script is particularly useful for transfer learning workflows, where a model trained on
a large dataset (like ImageNet) is fine-tuned for a specific task and then repurposed to
create rich feature representations for a new dataset.

Usage:
    - Ensure your fine-tuned model weights (`best_half_fine_tune.pth`), and your
      image datasets (`train_set.csv`, `val_set.csv`, `test_set.csv`) are in the
      same directory.
    - Run the script from the command line: `python resnet_feature_extractor.py`
    - The output will be three new CSV files containing the extracted features.

Dependencies:
    - torch
    - torchvision
    - pandas
    - Pillow (PIL)
    - numpy
    - tqdm
"""
import torch
import torch.nn as nn
import pandas as pd
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

# ----------------------------
# Load fine-tuned ResNet-18
# ----------------------------
class ResNet18FeatureExtractor(nn.Module):
    """
    A PyTorch module that loads a pre-trained ResNet-18 model and prepares it
    for feature extraction.

    The class performs the following steps:
    1. Initializes a ResNet-18 model with the specified number of output classes
       to match the trained configuration.
    2. Loads the fine-tuned weights from a `.pth` file.
    3. Replaces the final fully connected layer (`fc`) with an identity layer,
       which makes the model output the 512-dimensional features from the
       preceding layer instead of a classification result.
    4. Freezes all model parameters to prevent any further training, ensuring
       it acts solely as a feature extractor.

    Args:
        weights_path (str): The file path to the saved PyTorch model state dictionary.
    """
    def __init__(self, weights_path):
        super().__init__()
        # 1. Match your trained config (47 classes)
        model = models.resnet18(num_classes=47)

        # 2. Load trained weights
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict)

        # 3. Remove classifier to extract 512-dim features
        model.fc = nn.Identity()

        # 4. Freeze
        for param in model.parameters():
            param.requires_grad = False

        self.model = model
        self.eval()

    def forward(self, x):
        """
        Performs a forward pass to extract features.

        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The 512-dimensional feature vector.
        """
        return self.model(x)


# ----------------------------
# Image preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet input size
    transforms.ToTensor(),
])

def load_and_preprocess_image(path):
    """
    Loads an image from a given file path and applies the necessary
    preprocessing transformations.

    Args:
        path (str): The file path to the image.

    Returns:
        torch.Tensor: The preprocessed image as a PyTorch tensor.
    """
    image = Image.open(path).convert("RGB")
    return transform(image)

# ----------------------------
# Extract embeddings
# ----------------------------
def extract_embeddings(model, csv_path, output_file):
    """
    Extracts ResNet-18 embeddings for all images listed in a CSV file.

    This function reads a CSV file containing image paths, loads and
    processes each image, and uses the provided model to generate a
    512-dimensional feature vector. The resulting embeddings are saved
    along with their original labels into a new CSV file.

    Args:
        model (nn.Module): The feature extraction model (e.g., an instance of ResNet18FeatureExtractor).
        csv_path (str): The file path to the input CSV with image paths and labels.
        output_file (str): The file path where the output embeddings CSV will be saved.
    """
    df = pd.read_csv(csv_path)
    paths = df["image_path"].tolist()
    labels = df["label_idx"].tolist()

    features = []
    model.eval()
    for path in tqdm(paths):
        img = load_and_preprocess_image(path)
        img = img.unsqueeze(0)  # [1, 3, 224, 224]
        with torch.no_grad():
            vec = model(img)  # [1, 512]
        features.append(vec.squeeze(0).numpy())

    # Save as CSV
    out_df = pd.DataFrame(features)
    out_df["label"] = labels
    out_df.to_csv(output_file, index=False)
    print(f"Saved embeddings to {output_file}")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    model_path = "best_half_fine_tune.pth"  # path to your saved model
    extractor = ResNet18FeatureExtractor(model_path)

    extract_embeddings(extractor, "train_set.csv", "resnet18_train_embeddings_half_fine_tune.csv")
    extract_embeddings(extractor, "val_set.csv", "resnet18_val_embeddings_half_fine_tune.csv")
    extract_embeddings(extractor, "test_set.csv", "resnet18_test_embeddings_half_fine_tune.csv")