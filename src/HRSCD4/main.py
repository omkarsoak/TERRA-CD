from config import *
from dataloader import ChangeDetectionDatasetTIF, describe_loader
from torch.utils.data import DataLoader
import torch
from train import stage_one_training, stage_two_training, save_training_history
from models import HRSCD4
from test import test_model_complete, save_test_metrics
import os

def main():
   # Create datasets
    train_dataset = ChangeDetectionDatasetTIF(
        t2019_dir=f"{ROOT_DIRECTORY}/train/Images/T2019",
        t2024_dir=f"{ROOT_DIRECTORY}/train/Images/T2024",
        sem_2019_dir=f"{ROOT_DIRECTORY}/train/Masks/T2019",
        sem_2024_dir=f"{ROOT_DIRECTORY}/train/Masks/T2024",
        mask_dir=f"{ROOT_DIRECTORY}/train/{CD_DIR}",
        classes=CLASSES,
        semantic_classes=SEMANTIC_CLASSES
    )

    val_dataset = ChangeDetectionDatasetTIF(
        t2019_dir=f"{ROOT_DIRECTORY}/val/Images/T2019",
        t2024_dir=f"{ROOT_DIRECTORY}/val/Images/T2024",
        sem_2019_dir=f"{ROOT_DIRECTORY}/val/Masks/T2019",
        sem_2024_dir=f"{ROOT_DIRECTORY}/val/Masks/T2024",
        mask_dir=f"{ROOT_DIRECTORY}/val/{CD_DIR}",
        classes=CLASSES,
        semantic_classes=SEMANTIC_CLASSES
    )

    test_dataset = ChangeDetectionDatasetTIF(
        t2019_dir=f"{ROOT_DIRECTORY}/test/Images/T2019",
        t2024_dir=f"{ROOT_DIRECTORY}/test/Images/T2024",
        sem_2019_dir=f"{ROOT_DIRECTORY}/test/Masks/T2019",
        sem_2024_dir=f"{ROOT_DIRECTORY}/test/Masks/T2024",
        mask_dir=f"{ROOT_DIRECTORY}/test/{CD_DIR}",
        classes=CLASSES,
        semantic_classes=SEMANTIC_CLASSES
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print("------------Train------------")
    describe_loader(train_loader)
    print("------------Val------------")
    describe_loader(val_loader)
    print("------------Test------------")
    describe_loader(test_loader)


    ###### MODEL #########
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model configuration
    input_channels = 3
    num_semantic_classes = len(SEMANTIC_CLASSES)
    num_cd_classes = len(CLASSES)

    # Create model
    model = HRSCD4(
        input_channels=input_channels,
        num_semantic_classes=num_semantic_classes,
        num_cd_classes=num_cd_classes
    ).to(device)

    # Define checkpoint paths
    lcm_checkpoint_path = f'{SAVING_DIR}/best_lcm_model_{NUM_EPOCHS}.pt'
    full_checkpoint_path = f'{SAVING_DIR}/best_full_model_{NUM_EPOCHS}.pt'

    # Stage 1: Train LCM branches
    model1, stage1_history = stage_one_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_path=lcm_checkpoint_path,
        numepochs=NUM_EPOCHS
    )

    # Load best LCM model before stage 2
    if os.path.exists(lcm_checkpoint_path):
        checkpoint = torch.load(lcm_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best LCM model from epoch {checkpoint['epoch']}")

    # Stage 2: End-to-end training
    model2, stage2_history = stage_two_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_path=full_checkpoint_path,
        numepochs=NUM_EPOCHS
    )

    # Save training history
    save_path_stage1 = f'{SAVING_DIR}/strategy4_training_history_stage1_{NUM_EPOCHS}.json'
    save_path_stage2 = f'{SAVING_DIR}/strategy4_training_history_stage2_{NUM_EPOCHS}.json'
    save_path_bestepoch = f'{SAVING_DIR}/strategy4_best_epoch_{NUM_EPOCHS}.json'

    save_training_history(stage1_history,stage2_history, 
                        full_checkpoint_path, 
                        save_path_stage1, save_path_stage2, save_path_bestepoch)
    
    # Test the model
    test_metrics = test_model_complete(
        model=model,
        test_loader=test_loader,
        device=device,
        checkpoint_path=full_checkpoint_path,
        num_classes=num_cd_classes,
        num_semantic_classes=num_semantic_classes
    )
    save_path = f'{SAVING_DIR}/strategy4_test_metrics.json'
    save_test_metrics(test_metrics, save_path)
    
if __name__ == '__main__':
    main()