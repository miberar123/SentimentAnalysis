import numpy as np
import torch
import torch.nn as nn
import torchvision
from cnn import CNN, load_data
from transform_data import main as transform_data


def main():

    classification_models = torchvision.models.list_models(module=torchvision.models)

    # Load data and model

    train_dir = "./data/train"
    valid_dir = "./data/test"

    train_loader, valid_loader, num_classes = load_data(
        train_dir, valid_dir, batch_size=32, img_size=224
    )  # ResNet50 requires 224x224 images

    model = CNN(
        torchvision.models.resnet50(weights="DEFAULT"), num_classes, unfreezed_layers=2
    )

    classnames = train_loader.dataset.classes

    optimizer = torch.optim.Adam(model.base_model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    history = model.train_model(
        train_loader, valid_loader, optimizer, criterion, epochs=5
    )

    model.save("resnet50-5epochV3")

    predicted_labels = model.predict(valid_loader)
    true_labels = []
    for images, labels in valid_loader:
        true_labels.extend(labels.numpy())

    accuracy = np.mean(np.array(predicted_labels) == np.array(true_labels))
    print(f"Model Accuracy: {accuracy*100:.2f}%")


if __name__ == "__main__":
    transform_data()
    main()
