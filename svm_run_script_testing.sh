#!/bin/bash

echo "enter the cracked image directory"
read cracked_images

#echo "enter training - 0, testing - 1"
#read test

#echo "enter testing model"
#read model

ls "$cracked_images" > image_filenames.txt
mv features.txt features_bkp.txt
rm features.txt
num_files=$(cat image_filenames.txt | wc -l)
echo "$num_files"
counter=1
iter=1

#if [$test -eq 1]
#echo "enter testing model"
#read model
#fi
#echo "enter svm model file name"
#read svmModel
while [ $num_files -gt 0 ]
do
    firstArg=$(cat image_filenames.txt  | sed -n "$counter"p)
    counter=$((counter+1))
    secondArg=$(cat image_filenames.txt  | sed -n "$counter"p)
    counter=$((counter+1))
echo "iteration # $iter"
    ./test_bow_slic "$cracked_images" 1 "$firstArg" "$secondArg"

iter=$((iter+1))
#if [$test -eq 1]
#then
    rm features.txt
    mv output_svm.png svm_colorized_images/svm_out"$iter".png
    mv output_texton.png texton_map_images/texton_out"$iter".png
    mv texton_bw.png texton_map_bw_cracked/texton_bw"$iter".png
#fi
    let num_files=num_files-2
done
#if [$test -eq 0]
#then
#./libsvm-3.20/svm-train -s 0 -t 1 -d 5 -g 3 -r 1  features.txt featureModel.txt
#fi

