from config import *
from dataloader import ChangeDetectionDatasetTIF, describe_loader
from torch.utils.data import DataLoader
import torch
from train import train_strategy3, save_training_history
from models import HRSCD3
from test import test_strategy3, save_test_metrics

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_cd_classes= len(CLASSES)   
    num_semantic_classes= len(SEMANTIC_CLASSES)

    checkpoint_path = f'{SAVING_DIR}/best_Strat3_{NUM_EPOCHS}_epochs.pt'

    # Create model
    model = HRSCD3(
        cd_architecture='unet',
        lcm_architecture='unet',
        cd_encoder='resnet34',
        lcm_encoder='resnet34',
        input_channels=3,
        num_cd_classes=num_cd_classes,
        num_semantic_classes=num_semantic_classes
    ).to(device)

    # Train model
    model2, history = train_strategy3(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        device=device,
        checkpoint_path=checkpoint_path,
        num_cd_classes=num_cd_classes,
        num_semantic_classes=num_semantic_classes
    )

    save_path = f'{SAVING_DIR}/strat3_train_history.json'
    save_path_bestepoch = f'{SAVING_DIR}/strat3_bestepoch.json'
    save_training_history(history, checkpoint_path, save_path, save_path_bestepoch)

    ### Test model
    test_metrics = test_strategy3(model, test_loader, device,
                                    num_samples_to_plot=5,
                                    checkpoint_path=checkpoint_path,
                                    num_cd_classes=num_cd_classes,
                                    num_semantic_classes=num_semantic_classes)

    save_test_metrics(test_metrics, save_path=f'{SAVING_DIR}/strat3_test_metrics.json')



if __name__ == '__main__':
    main()