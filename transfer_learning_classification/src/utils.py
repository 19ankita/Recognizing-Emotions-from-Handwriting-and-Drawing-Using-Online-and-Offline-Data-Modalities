import torch

def accuracy(outputs, labels):
    preds = outputs.argmax(dim=1)
    return (preds == labels).float().mean().item()


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Loaded checkpoint from {path}")
