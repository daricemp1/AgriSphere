import torch
from torchvision import models, transforms
from PIL import Image

# Retrieves the description of the plant state from the technical class name 
def get_disease_description(class_name):
    # convert to lowercase for processing
    normalized = class_name.lower()
    
    # handle healthy cases
    if 'healthy' in normalized:
        return "Plant is healthy"
    
    # split by underscores and other separators
    parts = normalized.replace('-', '_').split('_')
    
    #turns this output into a readable sentences 
    if len(parts) >= 2:
        disease_parts = parts[1:]
        disease_name = ' '.join(disease_parts)
        disease_name = disease_name.title()        
        return f"{disease_name} detected"
    # if no disease is found , clean up and returns something readable 
    else:
        clean_name = normalized.replace('_', ' ').replace('-', ' ').title()
        return f"{clean_name} detected"

# Builds efficient net model and customizes the output layer 
def create_disease_model(num_disease_classes):
    model = models.efficientnet_b0(pretrained=True)
    #tailored to fit my number of disease classes 
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        num_disease_classes
    )
    return model

# 2. Load trained model and list of class names for inference
def load_vision_model():
    class_names = torch.load("class_names.pt")
    model = create_disease_model(len(class_names))
    model.load_state_dict(torch.load("efficientnet_disease.pt", map_location="cpu"))
    model.eval()
    # Debug: Print all available classes
    print("=== MODEL LOADING DEBUG ===")
    print(f"Total number of classes: {len(class_names)}")
    print("All available classes:")
    for i, class_name in enumerate(class_names):
        print(f"  {i}: {class_name}")
    print("=" * 30)
    
    return model, class_names

# 3. Define preprocessing (same as training), prepares the images so that model can understand them 
vision_transform = transforms.Compose([
    # resizes images 
    transforms.Resize((224, 224)),
    # turns it into a tensor which is pytorch's format 
    transforms.ToTensor(),
    # normalize pixel colors to match them with what efficient net expects 
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Takes an image and predicts the disease 
def predict_disease(image_path, model_bundle, return_confidence=False, debug=True):
    # unpacks the model and the class labels 
    model, class_names = model_bundle
    # opens the uploaded image and make sure that it is in colour
    img = Image.open(image_path).convert("RGB")
    # applies all the image processing steps and adds a batch dimension ( this is the batch size )
    tensor = vision_transform(img).unsqueeze(0)

    # Prediction 
    # runs the image through the model, but turns off training mode 
    with torch.no_grad():
        logits = model(tensor)
        # converts raw scores into probabilities 
        probabilities = torch.softmax(logits, dim=1)
        # finds the most likely class and the respective confidence score 
        predicted_idx = logits.argmax(dim=1).item()
        confidence = probabilities[0][predicted_idx].item()

        # get technical class name
        technical_class = class_names[predicted_idx]
        
        # auto-generate simple description
        simple_description = get_disease_description(technical_class)
        
        # Debugging information
        if debug:
            print(f"\n=== PREDICTION DEBUG for {image_path} ===")
            print(f"Predicted index: {predicted_idx}")
            print(f"Technical class predicted: '{technical_class}'")
            print(f"Generated description: '{simple_description}'")
            print(f"Confidence: {confidence:.4f}")
            
            # Show top 3 predictions for better debugging
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

# 5. Batch prediction function which runs prediction for multiple images at once 
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

# 6. Additional debugging function
def debug_class_mapping(class_names):
    """
    Debug function to see how each class name gets converted to description.
    """
    print("\n=== CLASS MAPPING DEBUG ===")
    for i, class_name in enumerate(class_names):
        description = get_disease_description(class_name)
        print(f"{i:2d}: '{class_name}' -> '{description}'")
    print("=" * 30)