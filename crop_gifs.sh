for i in `seq 1 5`
do
convert model1_evol_valimage_per_epoch$i.gif -coalesce -repage 0x0 -crop 700x400+50+600 +repage crop_model1_evol_valimage_per_epoch$i.gif
done
