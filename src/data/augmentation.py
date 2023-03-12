from torchvision import transforms


def training_transforms():
    return transforms.Compose(
        [
            transforms.RandomCrop(28, padding=2),
            transforms.RandomRotation(8),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
