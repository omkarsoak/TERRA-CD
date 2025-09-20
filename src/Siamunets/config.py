###### MODIFY ######
SAVING_DIR = "/scratch/home/"
ROOT_DIRECTORY = f"{SAVING_DIR}/CDMergedDividedSplit"
CD_DIR = "cd1_Output" # 'cd1_Output' or 'cd2_Output'
MODEL_NAME = 'siamunet_conc' #'siamunet_conc','siamunet_diff','siamunet_EF','snunet_conc','snunet_ECAM'
NUM_WORKERS = 8
BATCH_SIZE = 32

########## DO NOT MODIFY BELOW ##########
NUM_EPOCHS = 100

if CD_DIR == "cd1_Output":
    CLASSES = ['no_change','vegetation_increase','vegetation_decrease']
elif CD_DIR == "cd2_Output":
    CLASSES = ['no_change', 'water_building', 'water_sparse', 'water_dense',
               'building_water', 'building_sparse', 'building_dense',
               'sparse_water', 'sparse_building', 'sparse_dense',
               'dense_water', 'dense_building', 'dense_sparse']
