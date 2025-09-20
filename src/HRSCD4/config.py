ROOT_DIRECTORY = "CDMergedDividedSplit"
SAVING_DIR = "/scratch/home/"
CD_DIR = "cd2_Output"   #FOR STRATEGY4 ALWAYS USE cd2_Output

if CD_DIR == "cd1_Output":
    CLASSES = ['no_change','vegetation_increase','vegetation_decrease']
elif CD_DIR == "cd2_Output":
    CLASSES = ['no_change', 'water_building', 'water_sparse', 'water_dense',
               'building_water', 'building_sparse', 'building_dense',
               'sparse_water', 'sparse_building', 'sparse_dense',
               'dense_water', 'dense_building', 'dense_sparse']
    
SEMANTIC_CLASSES = ['water', 'building', 'sparse_vegetation', 'dense_vegetation']

NUM_WORKERS = 8
BATCH_SIZE = 32
NUM_EPOCHS = 100
MODEL_NAME = 'strat4'