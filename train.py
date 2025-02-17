import argparse
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os
from torchvision.models import VGG16_Weights


def get_args():
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset")
    parser.add_argument("data_directory", help="Directory containing the training data")
    parser.add_argument(
        "--save_dir", default="checkpoints", help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--arch", default="vgg16", choices=["vgg13", "vgg16"], help="Model architecture"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--hidden_units", type=int, default=512, help="Number of hidden units"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    return parser.parse_args()


def load_data(data_dir):
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
        "valid": datasets.ImageFolder(valid_dir, transform=data_transforms["valid"]),
    }

    dataloaders = {
        "train": torch.utils.data.DataLoader(
            image_datasets["train"], batch_size=64, shuffle=True
        ),
        "valid": torch.utils.data.DataLoader(image_datasets["valid"], batch_size=64),
    }

    return dataloaders, image_datasets


def build_model(arch, hidden_units):
    if arch == "vgg16":
        model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    else:
        model = models.vgg13(weights=VGG13_Weights.DEFAULT)

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define new classifier
    classifier = nn.Sequential(
        nn.Linear(25088, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1),
    )

    model.classifier = classifier
    return model


def train_model(model, dataloaders, criterion, optimizer, epochs, device):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for inputs, labels in dataloaders["train"]:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        model.eval()
        valid_loss = 0
        accuracy = 0

        with torch.no_grad():
            for inputs, labels in dataloaders["valid"]:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                valid_loss += criterion(outputs, labels).item()

                ps = torch.exp(outputs)
                top_ps, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(
            f"Epoch {epoch+1}/{epochs}.. "
            f"Train loss: {running_loss/len(dataloaders['train']):.3f}.. "
            f"Valid loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
            f"Valid accuracy: {accuracy/len(dataloaders['valid']):.3f}"
        )


def save_checkpoint(model, save_dir, arch, epochs, optimizer, image_datasets):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint = {
        "arch": arch,
        "epochs": epochs,
        "state_dict": model.state_dict(),
        "class_to_idx": image_datasets["train"].class_to_idx,
        "optimizer_state": optimizer.state_dict(),
    }

    torch.save(checkpoint, os.path.join(save_dir, f"{arch}_checkpoint.pth"))


def main():
    args = get_args()

    # Set device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Load data
    dataloaders, image_datasets = load_data(args.data_directory)

    # Build model
    model = build_model(args.arch, args.hidden_units)

    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # Train model
    train_model(model, dataloaders, criterion, optimizer, args.epochs, device)

    # Save checkpoint
    save_checkpoint(
        model, args.save_dir, args.arch, args.epochs, optimizer, image_datasets
    )


if __name__ == "__main__":
    main()
