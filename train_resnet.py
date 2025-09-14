import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import os
import time
from tqdm import tqdm

def evaluate_model(model, dataloader, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(model.device), labels.to(model.device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds, target_names=class_names)

    # Log to file
    with open("model_eval_log.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(" Evaluation metrics saved to model_eval_log.txt")
    return accuracy

def main():
    data_dir = "sample_data/images"
    batch_size = 32  
    epochs = 5
    model_out = "resnet_disease.pt"
    labels_out = "class_names.pt"

    # Check device availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  WARNING: Training on CPU will be very slow!")
        print("Consider using Google Colab or a machine with GPU")

    # Optimized transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    class_names = full_dataset.classes
    train_len = int(0.8 * len(full_dataset))
    val_len = len(full_dataset) - train_len
    train_ds, val_ds = random_split(full_dataset, [train_len, val_len])
    
    # Apply different transforms to validation set
    val_ds.dataset.transform = val_transform

    print(f" Starting training on {len(class_names)} classes")
    print(f" Dataset: {train_len} train, {val_len} validation samples")
    print(f" Batches per epoch: {train_len // batch_size}")

    # Optimized data loaders
    num_workers = 4 if device.type == 'cuda' else 2
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Model setup
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))


    model.to(device)
    model.device = device

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    # Training loop with progress tracking
    best_val_acc = 0.0
    train_losses = []
    
    print(" Starting training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\n Epoch {epoch+1}/{epochs}")
        print("-" * 30)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0
        
        # Use tqdm for progress bar
        train_pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
        
        for batch_idx, (imgs, labels) in enumerate(train_pbar):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_preds += (predicted == labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.0 * correct_preds / total_samples:.2f}%'
            })
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct_preds / total_samples
        train_losses.append(epoch_loss)
        
        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for imgs, labels in val_pbar:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'Val Loss': f'{loss.item():.4f}',
                    'Val Acc': f'{100.0 * val_correct / val_total:.2f}%'
                })
        
        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        
        print(f" Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_{model_out}")
            print(f"New best model saved! Val Acc: {val_acc:.2f}%")
        
        # Update learning rate
        scheduler.step()
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Training completed
    total_time = time.time() - start_time
    print(f"\n Training completed in {total_time/60:.1f} minutes")
    print(f" Best validation accuracy: {best_val_acc:.2f}%")

    # Save final model
    torch.save(model.state_dict(), model_out)
    torch.save(class_names, labels_out)
    print(f" Final model saved to {model_out}")
    print(f" Labels saved to {labels_out}")

    # Final evaluation
    print("\n Running final evaluation...")
    final_acc = evaluate_model(model, val_loader, class_names)
    print(f"Final validation accuracy: {final_acc:.4f}")

if __name__ == "__main__":
    main()