

################# B
python3 train.py --num_epochs 200 --dataset Geomorph_B --num_val_images 58 --model custom --batch_size 2


mv accuracy_vs_epochs.png Geomorph_B/B_accuracy_vs_epochs.png
mv iou_vs_epochs.png Geomorph_B/B_iou_vs_epochs.png
mv loss_vs_epochs.png Geomorph_B/B_loss_vs_epochs.png

for k in `seq 0 9`
do
#rm -rf "checkpoints/000"$k
mv checkpoints/000$k checkpoints/Geomorph_B
done 

for k in `seq 10 99`
do
#rm -rf "checkpoints/00"$k
mv checkpoints/00$k checkpoints/Geomorph_B
done 

for k in `seq 100 198`
do
#rm -rf "checkpoints/0"$k
mv checkpoints/0$k checkpoints/Geomorph_B
done 
 

#mv checkpoints/0199 checkpoints/Geomorph_B
mv checkpoints/checkpoint checkpoints/Geomorph_B
mv checkpoints/*.ckpt.* checkpoints/Geomorph_B



##python3 predict.py --model custom --dataset Geomorph_B --image Geomorph_B/val/B_tile_9728_1536.jpg --checkpoint_path checkpoints/Geomorph_B/latest_model_custom_Geomorph_B.ckpt

for file in Geomorph_B/val/*.jpg
do
python3 predict.py --model custom --dataset Geomorph_B --image $file --checkpoint_path checkpoints/Geomorph_B/0190/model.ckpt
done
