import argparse
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import json
import torch.nn as nn
from torchvision.models import VGG16_Weights, VGG13_Weights


def get_args():
    parser = argparse.ArgumentParser(description="Predict flower name from an image")
    parser.add_argument("input", help="Path to image")
    parser.add_argument("checkpoint", help="Path to checkpoint")
    parser.add_argument(
        "--top_k", type=int, default=5, help="Return top K most likely classes"
    )
    parser.add_argument(
        "--category_names", help="Path to JSON file mapping categories to real names"
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")
    return parser.parse_args()


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, weights_only=True)

    # Create the same model architecture used in training
    if checkpoint["arch"] == "vgg16":
        model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    else:
        model = models.vgg13(weights=VGG13_Weights.DEFAULT)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Build classifier with same architecture as training
    classifier = nn.Sequential(
        nn.Linear(25088, 512),  # Using default hidden_units=512 from train.py
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 102),  # 102 flower classes
        nn.LogSoftmax(dim=1),
    )

    model.classifier = classifier
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]

    return model


def process_image(image_path):
    img = Image.open(image_path)

    # Resize
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))

    # Crop
    left = (img.width - 224) / 2
    top = (img.height - 224) / 2
    right = left + 224
    bottom = top + 224
    img = img.crop((left, top, right, bottom))

    # Convert to numpy and normalize
    np_image = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Transpose
    np_image = np_image.transpose((2, 0, 1))

    return torch.from_numpy(np_image).type(torch.FloatTensor)


def predict(image_path, model, device, topk=5):
    model.to(device)
    model.eval()

    # Process image
    img = process_image(image_path)
    img = img.unsqueeze(0).to(device)

    # Calculate probabilities
    with torch.no_grad():
        output = model(img)
        probs = torch.exp(output)
        top_probs, top_indices = probs.topk(topk)

    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = []

    # Safely convert indices to classes
    for idx in top_indices[0]:
        idx_val = idx.item()
        if idx_val in idx_to_class:
            top_classes.append(idx_to_class[idx_val])
        else:
            print(f"Warning: Index {idx_val} not found in class mapping")
            top_classes.append(str(idx_val))

    return top_probs[0].cpu().numpy(), top_classes


def main():
    args = get_args()

    # Set device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Load model
    model = load_checkpoint(args.checkpoint)

    # Load category names
    if args.category_names:
        with open(args.category_names, "r") as f:
            cat_to_name = json.load(f)
    else:
        cat_to_name = None

    # Make prediction
    probs, classes = predict(args.input, model, device, args.top_k)

    # Print results
    for i in range(len(probs)):
        if cat_to_name:
            print(f"{cat_to_name[classes[i]]}: {probs[i]*100:.2f}%")
        else:
            print(f"Class {classes[i]}: {probs[i]*100:.2f}%")


if __name__ == "__main__":
    main()
