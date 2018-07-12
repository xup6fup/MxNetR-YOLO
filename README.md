
YOLO v3 in MxNetR
===

This is a simple example for implementing the YOLO by MxNetR. The idea is devoloped by Joseph Chet Redmon, and related details can be found in his [website](Joseph Chet Redmon). This is a simple example for MxNetR user, I will use a relatively small dataset for demonstrating how it work. 

Download dataset and pre-processing
---

The Pascal VOC challenge is a very popular dataset for building and evaluating algorithms for image classification, object detection, and segmentation. I will use the mirror website for downloading **VOC2007** dataset. You can use the code ["1. download VOC2007.R"](https://github.com/xup6fup/MxNetR-YOLO/blob/master/code/Processing%20data/1.%20download%20VOC2007.R) to quickly download this dataset (439 MB for training and 431 MB for testing). **Note: if you just want to use pre-trained model, you don't need to download this dataset.**

To simplify the problem, we will resize all images as 256×256. You can use the codes ["2-1. pre-processing image (train & val).R"](https://github.com/xup6fup/MxNetR-YOLO/blob/master/code/Processing%20data/2-1.%20pre-processing%20image%20(train%20%26%20val).R) and ["2-2. pre-processing image (test).R"](https://github.com/xup6fup/MxNetR-YOLO/blob/master/code/Processing%20data/2-2.%20pre-processing%20image%20(test).R) to do this work. Resized images will be converted as .RData and they totally used about 15GB for storing them.

After we get the bounding box infomation in pevious stage, we can caculate the anchor boxes by k-mean clustering analysis. In yolo v3, there are 9 anchor boxes belonging feature map with 8 stride (3 smallest), 16 stride (3 meddle size), and 32 stride (3 biggest), respectively. You can use the codes ["3. Define the anchor boxes.R"](https://github.com/xup6fup/MxNetR-YOLO/blob/master/code/Processing%20data/3.%20Define%20the%20anchor%20boxes.R) for conducting this process. Finally, we will get the **anchor_boxs.RData** for further application.

Training stage
---

The first step for using MxNet to train a yolo model is to build an iterator. You can use the codes ["1. Encode, Decode & Iterator.R"](https://github.com/xup6fup/MxNetR-YOLO/blob/master/code/Training/1.%20Encode,%20Decode%20%26%20Iterator.R) for conducting this process. It is worth noting that bounding boxes are needed to encode as a special form for following training. Moreover, the encoded labels also need to pass a decoding process for restoring bounding boxes. The encode and decode function are the core of the yolo model. If you want to clearly understand the principle of yolo model, you can dismantle these functions to learn. The test codes for generating images are also included in that code, let's try it!


 



