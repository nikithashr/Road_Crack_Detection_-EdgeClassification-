#/!bin/bash
iter=1
while [ $iter -lt 10 ]
do
echo $iter
./test_bow_slic train_cracks/cracked_road_cloudy 0 20130719_163110_833_004.jpg 20130719_163110_833_004_mask.png 1
iter=$((iter+1))
done

