import torch
from torchvision import models, transforms
from PIL import Image

# Converts technical class name into readable description
def get_disease_description(class_name):
    normalized = class_name.lower()

    if 'healthy' in normalized:
        return "Plant is healthy"

    parts = normalized.split('_')

    if len(parts) >= 2:
        crop = parts[0].capitalize()
        disease = ' '.join([p.capitalize() for p in parts[1:]])
        return f"{crop} - {disease} detected"
    else:
        clean_name = normalized.replace('_', ' ').title()
        return f"{clean_name} detected"

# Builds EfficientNet model with custom classifier layer
def create_disease_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

# Loads trained model and class names
def load_vision_model():
    class_names = torch.load("class_names.pt")
    model = create_disease_model(len(class_names))
    model.load_state_dict(torch.load("resnet_disease.pt", map_location="cpu"))
    model.eval()
    return model, class_names


    # Debug: Print all available classes
    print("=== MODEL LOADING DEBUG ===")
    print(f"Total number of classes: {len(class_names)}")
    for i, class_name in enumerate(class_names):
        print(f"  {i}: {class_name}")
    print("=" * 30)

    return model, class_names

# Preprocessing pipeline
vision_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Predict disease from a single image
def predict_disease(image_path, model_bundle, return_confidence=False, debug=True):
    model, class_names = model_bundle
    img = Image.open(image_path).convert("RGB")
    tensor = vision_transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_idx = logits.argmax(dim=1).item()
        confidence = probabilities[0][predicted_idx].item()

        technical_class = class_names[predicted_idx]
        simple_description = get_disease_description(technical_class)

        if debug:
            print(f"\n=== PREDICTION DEBUG for {image_path} ===")
            print(f"Predicted index: {predicted_idx}")
            print(f"Technical class predicted: '{technical_class}'")
            print(f"Generated description: '{simple_description}'")
            print(f"Confidence: {confidence:.4f}")
            top_3_indices = torch.topk(probabilities[0], 3).indices
            top_3_probs = torch.topk(probabilities[0], 3).values
            print("Top 3 predictions:")
            for i, (idx, prob) in enumerate(zip(top_3_indices, top_3_probs)):
                class_name = class_names[idx.item()]
                description = get_disease_description(class_name)
                print(f"  {i+1}. {class_name} -> '{description}' (confidence: {prob:.4f})")
            print("=" * 50)

        if return_confidence:
            return simple_description, confidence, technical_class
        else:
            return simple_description

# Predict multiple images in a batch
def predict_diseases_batch(image_paths, model_bundle, return_confidence=False, debug=True):
    results = []
    for image_path in image_paths:
        try:
            result = predict_disease(image_path, model_bundle, return_confidence, debug)
            results.append(result)
        except Exception as e:
            error_msg = f"Error processing {image_path}: {str(e)}"
            if debug:
                print(f"ERROR: {error_msg}")
            if return_confidence:
                results.append((error_msg, 0.0, "error"))
            else:
                results.append(error_msg)
    return results

# Utility for debugging class name conversion
def debug_class_mapping(class_names):
    print("\n=== CLASS MAPPING DEBUG ===")
    for i, class_name in enumerate(class_names):
        description = get_disease_description(class_name)
        print(f"{i:2d}: '{class_name}' -> '{description}'")
    print("=" * 30)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_model_on_dataset(model, dataloader, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print("\n=== MODEL EVALUATION METRICS ===")
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}\n")

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Optionally visualize confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=90)
    plt.tight_layout()
    plt.show()
