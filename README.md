
YOLO in MxNetR
===

This is a simple example for implementing the YOLO by MxNetR. The idea is devoloped by Joseph Chet Redmon, and related details can be found in his [website](Joseph Chet Redmon). This is a simple example for MxNetR user, I will use a relatively small dataset for demonstrating how it work. 

Download dataset and pre-processing
---

The Pascal VOC challenge is a very popular dataset for building and evaluating algorithms for image classification, object detection, and segmentation. I will use the mirror website for downloading **VOC2007** dataset. You can use the code ["1. download VOC2007.R"](https://github.com/xup6fup/MxNetR-YOLO/blob/master/code/1.%20download%20VOC2007.R) to quickly download this dataset (439 MB for training and 431 MB for testing). **Note: if you just want to use pre-trained model, you don't need to download this dataset.**

To simplify the problem, we will resize all images as 256*256. You can use the codes ["2-1. pre-processing image (train & val).R"](https://github.com/xup6fup/MxNetR-YOLO/blob/master/code/2-1.%20pre-processing%20image%20(train%20%26%20val).R) and ["2-2. pre-processing image (test).R"](https://github.com/xup6fup/MxNetR-YOLO/blob/master/code/2-2.%20pre-processing%20image%20(test).R) to do this work. Resized images will be converted as .RData and they totally used about 15GB for storing them.


 



