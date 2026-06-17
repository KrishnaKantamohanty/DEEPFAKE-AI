import os
import argparse
from PIL import Image
import torch
from torchvision import transforms

# Import model loader
from model import get_model

def preprocess_image(image_path):
    """
    Loads an image, converts it to RGB, resizes it to 224x224,
    converts to a tensor, and normalizes it using ImageNet stats.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path '{image_path}' does not exist.")

    # Open image and ensure RGB mode (forces 3 channels, drops alpha channel if present)
    image = Image.open(image_path).convert('RGB')
    
    # Preprocessing pipeline identical to validation loader
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    image_tensor = preprocess(image)
    # Add a batch dimension: [3, 224, 224] -> [1, 3, 224, 224]
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def predict_details(image_path, model_path='best_model.pth'):
    """
    Loads model checkpoint, runs prediction on the image,
    and returns class prediction, confidence score, and raw fake probability.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Instantiate the model topology without downloading ImageNet weights.
    # The trained checkpoint fully defines the inference weights.
    model = get_model(fine_tune=False, device=device, pretrained=False)
    
    # Load trained model checkpoint weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model checkpoint '{model_path}' not found. Please train the model first.")
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval() # Set model to evaluation mode (deactivates Dropout/BatchNorm update)

    # Preprocess image
    image_tensor = preprocess_image(image_path).to(device)
    
    # Run forward pass (disable gradient calculation for inference speed and memory savings)
    with torch.no_grad():
        logits = model(image_tensor).squeeze(1) # Squeeze output to scalar logit
        probability = torch.sigmoid(logits).item() # Map raw logit to probability range [0, 1]

    # Convert probability to class and confidence score
    # label 0 = REAL, label 1 = FAKE
    if probability >= 0.5:
        prediction = "FAKE/AI-GENERATED"
        confidence = probability * 100
    else:
        prediction = "REAL/AUTHENTIC"
        confidence = (1.0 - probability) * 100
        
    return {
        'prediction': prediction,
        'confidence': confidence,
        'fake_probability': probability * 100,
        'real_probability': (1.0 - probability) * 100
    }


def predict(image_path, model_path='best_model.pth'):
    """
    Backward-compatible prediction helper.
    """
    details = predict_details(image_path, model_path=model_path)
    return details['prediction'], details['confidence']

def main():
    parser = argparse.ArgumentParser(description="AI Fake Image Prediction - Single-Image Inference")
    parser.add_argument('--image_path', type=str, required=True, 
                        help="Path to the new, unseen image file to be analyzed.")
    parser.add_argument('--model_path', type=str, default='best_model.pth', 
                        help="Path to the trained model file. Defaults to 'best_model.pth'.")
    args = parser.parse_args()

    try:
        print(f"Analyzing image: {args.image_path}")
        details = predict_details(args.image_path, args.model_path)
        
        print("\n" + "="*50)
        print("INFERENCE RESULT")
        print("="*50)
        print(f"Classification: {details['prediction']}")
        print(f"Confidence:     {details['confidence']:.2f}%")
        print(f"Real score:     {details['real_probability']:.2f}%")
        print(f"Fake score:     {details['fake_probability']:.2f}%")
        print("="*50)
        
    except Exception as e:
        print(f"\nError during inference: {e}")

if __name__ == '__main__':
    main()
