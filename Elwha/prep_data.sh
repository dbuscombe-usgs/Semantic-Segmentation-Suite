
mkdir val
mkdir val_labels
mkdir test
mkdir test_labels
mkdir train
mkdir train_labels
mkdir label_tiles
mkdir R_G_B_tiles

##split rasters into small tiles
python3 split_rasters_labels.py 
python3 split_rasters.py

# randomly split imagery into categories
python3 split_test_val_train.py

python3 list_codes.py
