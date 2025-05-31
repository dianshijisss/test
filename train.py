import argparse
import torch
from torch import nn
from torch import optim
from dataset import get_dataloaders
from model import create_model


def train(args):
    train_loader, val_loader, num_classes = get_dataloaders(args.train_dir, args.val_dir, args.batch_size)
    model = create_model(num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

    torch.save(model.state_dict(), args.output)
    print('Model saved to', args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train skin disease detector')
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--val-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output', type=str, default='model.pth')
    args = parser.parse_args()
    train(args)
