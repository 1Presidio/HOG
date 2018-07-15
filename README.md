# HOG

This is a manual C++ implementation of the classical HOG classifier. See the write-up for more details.

To run the code simply run the following commands

```
    $ bash run.sh MOVIE_PATH
```

Dependencies include *ffmpeg, openCV2, glog (logging), tcmalloc (debugger), eigen, boost

Source Paper: http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf

Additional Sources:

    https://www.learnopencv.com/histogram-of-oriented-gradients/

    http://mccormickml.com/2013/05/09/hog-person-detector-tutorial/

Data sets pulled from:

    http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/index.html

    http://www.gavrila.net/Datasets/Daimler_Pedestrian_Benchmark_D/daimler_pedestrian_benchmark_d.html