from config import *
from dataloader import ChangeDetectionDatasetTIF, describe_loader
from torch.utils.data import DataLoader
import torch
from train import train_model_balanced, save_training_files
from models import SiamUnet_conc, SiamUnet_diff, SiamUnet_EF, Siam_NestedUNet_Conc, SNUNet_ECAM
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
    
    ###### MODEL RUN #########
    # Initialize and train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    strategy = 'st2'
    num_cd_classes = len(CLASSES)  #num classes in change mask
    weighting_method = 'square_balanced'  #balanced,square_balanced,custom

    weighted_metrics = True if num_cd_classes > 5 else False   #True for 13 classes,False for 3 classes

    name = f"{strategy}_{MODEL_NAME}-{num_cd_classes}_classes_{NUM_EPOCHS}"
    checkpoint_path = f'{SAVING_DIR}/best_{name}.pt'

    if MODEL_NAME == 'siamunet_conc':
        model = SiamUnet_conc(input_nbr=3, label_nbr=num_cd_classes).to(device)
    elif MODEL_NAME == 'siamunet_diff':
        model = SiamUnet_diff(input_nbr=3, label_nbr=num_cd_classes).to(device)
    elif MODEL_NAME == 'siamunet_EF':
        model = SiamUnet_EF(input_nbr=3, label_nbr=num_cd_classes).to(device)
    elif MODEL_NAME == 'snunet_conc':
        model = Siam_NestedUNet_Conc(in_ch=3, out_ch=num_cd_classes).to(device)
    elif MODEL_NAME == 'snunet_ECAM':
        model = SNUNet_ECAM(in_ch=3, out_ch=num_cd_classes).to(device)

    model2, history = train_model_balanced(model, train_loader, val_loader,
                                        num_epochs=NUM_EPOCHS, num_cd_classes=num_cd_classes,
                                        device=device,
                                        weighting_method=weighting_method,
                                        checkpoint_path=checkpoint_path,
                                        weighted_metrics=weighted_metrics)


    history_filename = f"{SAVING_DIR}/{name}_history.json"
    bestepoch_filename = f"{SAVING_DIR}/{name}_best_epoch.json"
    save_training_files(history=history,checkpoint_path=checkpoint_path,
                        history_filename=history_filename,bestepoch_filename=bestepoch_filename)
    

    # Test the model
    checkpoint_path = f'{SAVING_DIR}/best_{name}.pt'
    random_samples, test_metrics = test_model(model, test_loader, device=device,
                                            num_cd_classes=num_cd_classes,
                                            weighted_metrics=weighted_metrics,
                                            checkpoint_path=checkpoint_path)

    # Save test metrics
    save_test_metrics(test_metrics=test_metrics,
                    save_path=f'{SAVING_DIR}/{name}_test_metrics.json')

    # Visualize results
    visualize_results(random_samples,num_cd_classes=num_cd_classes)
    
if __name__ == '__main__':
    main()