import argparse
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import create_model


def visualize(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    class_names = dataset.classes

    model = create_model(args.num_classes)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    images, labels = next(iter(loader))
    images = images[: args.num_images]
    labels = labels[: args.num_images]

    with torch.no_grad():
        outputs = model(images.to(device))
        _, preds = torch.max(outputs, 1)

    images = images.cpu()
    fig, axes = plt.subplots(1, len(images), figsize=(len(images) * 3, 3))
    if len(images) == 1:
        axes = [axes]
    for ax, img, label, pred in zip(axes, images, labels, preds):
        img = img.permute(1, 2, 0)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'True: {class_names[label]}\nPred: {class_names[pred]}')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize model predictions')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--num-classes', type=int, required=True)
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with images')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-images', type=int, default=4)
    args = parser.parse_args()
    visualize(args)
