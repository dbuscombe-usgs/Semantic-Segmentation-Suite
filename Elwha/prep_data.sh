
mkdir val
mkdir val_labels
mkdir test
mkdir test_labels
mkdir train
mkdir train_labels
mkdir label_tiles
mkdir R_G_B_tiles

##split rasters into small tiles
python Elwha/split_rasters_labels.py 
python Elwha/split_rasters.py

# randomly split imagery into categories
python Elwha/split_test_val_train.py

python Elwha/list_codes.py
