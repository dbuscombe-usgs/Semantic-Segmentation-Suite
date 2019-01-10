##model 1:
#python3 train.py --num_epochs 300 --dataset Geomorph --num_val_images 147
#(FC-DenseNet56 and ResNet101)

##model 2:
#python3 train.py --num_epochs 300 --dataset Geomorph --num_val_images 147 --frontend InceptionV4 --model FC-DenseNet103

##model 3:
python3 train.py --num_epochs 300 --dataset Geomorph --num_val_images 147 --frontend InceptionV4 --model DeepLabV3

#--h_flip True --v_flip True --brightness 0.1 --rotation 30 

#python3 test.py --checkpoint_path checkpoints/ --dataset Geomorph --model FC-DenseNet103

