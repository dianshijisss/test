import argparse
import torch
from PIL import Image
from torchvision import transforms
from model import create_model


def predict(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(args.num_classes)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(args.image).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)
    print('Predicted class:', predicted.item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict skin disease class')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--num-classes', type=int, required=True)
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()
    predict(args)
