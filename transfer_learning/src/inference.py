import torch
from torchvision import transforms
from PIL import Image
import argparse
from model import build_resnet18

def load_image(img_path, img_size=224):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert("RGB")
    return tf(img).unsqueeze(0)

def main(image_path, checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_resnet18(num_classes=3, freeze_backbone=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()

    img = load_image(image_path).to(device)

    with torch.no_grad():
        outputs = model(img)
        pred = outputs.argmax(dim=1).item()

    print(f"Predicted class: {pred}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    main(args.image, args.checkpoint)
