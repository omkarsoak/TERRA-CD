from config import *
from dataloader import ChangeDetectionDatasetTIF, describe_loader
from torch.utils.data import DataLoader
import torch
from train import train_model, save_training_files
from models import STANet
from test import test_model, save_test_metrics, visualize_results

def main():
    # Create datasets
    train_dataset = ChangeDetectionDatasetTIF(
        t2019_dir=f"{ROOT_DIRECTORY}/train/Images/T2019",
        t2024_dir=f"{ROOT_DIRECTORY}/train/Images/T2024",
        mask_dir=f"{ROOT_DIRECTORY}/train/{CD_DIR}",
        classes=CLASSES
    )

    val_dataset = ChangeDetectionDatasetTIF(
        t2019_dir=f"{ROOT_DIRECTORY}/val/Images/T2019",
        t2024_dir=f"{ROOT_DIRECTORY}/val/Images/T2024",
        mask_dir=f"{ROOT_DIRECTORY}/val/{CD_DIR}",
        classes=CLASSES
    )

    test_dataset = ChangeDetectionDatasetTIF(
        t2019_dir=f"{ROOT_DIRECTORY}/test/Images/T2019",
        t2024_dir=f"{ROOT_DIRECTORY}/test/Images/T2024",
        mask_dir=f"{ROOT_DIRECTORY}/test/{CD_DIR}",
        classes=CLASSES
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=NUM_WORKERS, pin_memory=True)

    print("------------Train-----------")
    describe_loader(train_loader)
    print("------------Val------------")
    describe_loader(val_loader)
    print("------------Test------------")
    describe_loader(test_loader)

    # Initialize and train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'stanet'
    attention_mode = ATTENTION_NODE
    num_classes = 3  #num classes in change mask
    weighting_method = 'square_balanced' #'custom'
    loss = 'CE' #'focal' #'bcl'
    checkpoint_path = f'{SAVING_DIR}/best_{model_name}_{attention_mode}-{num_classes}_classes_{NUM_EPOCHS}.pt'

    # Initialize model and data loaders
    model = STANet(input_channels=3, hidden_channels=32,
                num_classes=num_classes,
                attention_mode=attention_mode)

    # Train model
    model, history = train_model(
        model=model,
        train_loader=train_loader,val_loader=val_loader,
        num_epochs=NUM_EPOCHS,num_classes=num_classes,
        device=device,
        learning_rate=1e-4,weight_decay=0.01,
        checkpoint_path=checkpoint_path
    )

    history_filename = f"{SAVING_DIR}/{model_name}_{attention_mode}-{num_classes}_classes_{NUM_EPOCHS}_history.json"
    bestepoch_filename = f"{SAVING_DIR}/{model_name}_{attention_mode}-{num_classes}_classes_{NUM_EPOCHS}_best_epoch.json"
    save_training_files(history=history,checkpoint_path=checkpoint_path,
                        history_filename=history_filename,bestepoch_filename=bestepoch_filename)
    
    # Test the model
    random_samples, test_metrics = test_model(model, test_loader, loss=loss,
                                            device=device, num_classes=num_classes,
                                            checkpoint_path=checkpoint_path)

    # Save test metrics
    save_test_metrics(test_metrics=test_metrics,
                    save_dir=SAVING_DIR,
                    model_name=model_name,
                    attention_mode=attention_mode,
                    num_epochs=NUM_EPOCHS, num_classes=num_classes)

    # Visualize results
    visualize_results(random_samples,num_classes=num_classes)

if __name__ == "__main__":
    main()