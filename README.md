To run the code:
<p> a) cmake . </p>
<p> b) make </p>
<p> c) ./test_bow_slic '0/1 - training/testing' 'path to image' 'path to mask image' 'label for svm training' </p>
<br> OR after make, run:
<p> d) ./svm_run_training.sh  </p>
<br> run svm_run_training.sh with both positive and negative examples
<p> e) ./svm_run_training.sh </p>
<p> e) ./libsvm-3.20/svm-train -s 0 -t 1 -d 5 -g 3 -r 1  features.txt featureModel.txt </p>
<p> note: run make in ./libsvm-3.20 if you are running the above command for the first time in the directory </p>
<p> f) ./svm_run_testing.sh </p>
<p> g) To change the number of kmeans centers, goto bow_slic.hpp and change "#define KMEANS " to the desired value.
<p> h) To change the filter_size for texton map generation, change #define FILTER_SIZE " </p>
<p> i) note: make sure you have an updated version of the file "kmeans.txt" with the right number of centers and filter size </p>
<p> j) To compute kmeans.txt, go here: https://github.com/nikithashr/Texton-Map-Generation- </p>
