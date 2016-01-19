#!/bin/bash

echo "enter the cracked image directory"
read cracked_images

echo "create empty folders svm_image_colorized,  texton_map_images, texton_map_bw"

#echo "press 1 for testing"
#read test

#echo "enter testing model"
#read model

ls "$cracked_images" > image_filenames.txt

echo "backing up features.txt before beginning testing"
mv features.txt features_bkp.txt
rm features.txt

rm -rf svm_colorized_images
mkdir svm_colorized_images

rm -rf texton_map_images
mkdir texton_map_images

num_files=$(cat image_filenames.txt | wc -l)
echo "$"
counter=1
iter=1

#if [$test -eq 1]
#echo "enter testing model"
#read model
#fi
#echo "enter svm model file name"
#read svmModel

#arbitrary label assignment for testing - doesn't matter what it is
#label_svm = 1

while [ $num_files -gt 0 ]
do
    firstArg=$(cat image_filenames.txt  | sed -n "$counter"p)
    counter=$((counter+1))
    secondArg=$(cat image_filenames.txt  | sed -n "$counter"p)
    counter=$((counter+1))
echo "iteration # $iter"
    ./test_bow_slic "$cracked_images" 1 "$firstArg" "$secondArg" 1

iter=$((iter+1))
#if [$test -eq 1]
#then
    rm features.txt
    mv output_svm.png svm_colorized_images/svm_out"$iter".png
    mv output_texton.png texton_map_images/texton_out"$iter".png
    mv texton_bw.png texton_map_bw/texton_bw"$iter".png
#fi
    let num_files=num_files-2
done
#if [$test -eq 0]
#then
#./libsvm-3.20/svm-train -s 0 -t 1 -d 5 -g 3 -r 1  features.txt featureModel.txt
#fi

