# SLIC Superpixel Implementation
This repository contains an implementation of the SLIC Superpixel algorithm by Achanta et al. (PAMI'12, vol. 34, num. 11, pp. 2274-2282). The C++ implementation is created to work with the strutures of OpenCV.

## Exemplary result
The images below shows an example of an over-segmentation using 400 superpixels and a weight factor of 40.
<p align="center">
  <img src="https://github.com/PSMM/SLIC-Superpixels/blob/master/dog.png?raw=true" alt="Dog"/>
  <img src="https://github.com/PSMM/SLIC-Superpixels/blob/master/dog_segmentation.png?raw=true" alt="Dog Segmentation"/>
</p>

3). To run the code:
<p> a) cmake . </p>
<p> b) make </p>
<p> c) ./test_bow_slic images/street_image.jpg 2000 40 60 15 </p>
<p> d) To change the number of kmeans centers, goto bow_slic.hpp and change "#define KMEANS " to the desired value.
       To change the filter_size for texton map generation, change #define FILTER_SIZE " </p>
<p> note: make sure you have an updated version of the file "kmeans.txt" with the right number of centers and filter size </p>
<p> e) To compute kmeans.txt, go here: https://github.com/nikithashr/Texton-Map-Generation- </p>
