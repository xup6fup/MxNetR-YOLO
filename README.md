
YOLO v3 in MxNetR
===

This is a simple example for implementing the YOLO by MxNetR. The idea is devoloped by Joseph Chet Redmon, and related details can be found in his [website](https://pjreddie.com/darknet/yolo/?utm_source=next.36kr.com). This is a simple example for MxNetR user, I will use a relatively small dataset for demonstrating how it work. 

# If you just want to use this model for predicting

You can use the code ["1. Prediction.R"](https://github.com/xup6fup/MxNetR-YOLO/blob/master/code/3.%20Predicting/1.%20Prediction.R) for predicting an image. You need to download [yolo_v3-0000.params](https://drive.google.com/open?id=1NDCaWLQev43K3pqhCSYVLNo2FVpVqgh_) and [yolo_v3-symbol.json](https://drive.google.com/open?id=1unzvMu0hKMLi2gmeLywqnXBguAUWQoZb) and put them in the folder 'model/yolo model' Please click my above superlink for download. 

Here we use the 'test_img.jpeg' for testing the model. The left image is the raw image, and the right one is the prediction result by yolo v3 model.

<p align="center">
  <img src="pred_test_img.jpeg">
</p>

Let try to predict other image!

# If you want to train a yolo v3 model

1. Download dataset and pre-processing
---

The Pascal VOC challenge is a very popular dataset for building and evaluating algorithms for image classification, object detection, and segmentation. I will use the mirror website for downloading **VOC2007** dataset. You can use the code ["1. download VOC2007.R"](https://github.com/xup6fup/MxNetR-YOLO/blob/master/code/1.%20Processing%20data/1.%20download%20VOC2007.R) to quickly download this dataset (439 MB for training and 431 MB for testing). **Note: if you just want to use pre-trained model, you don't need to download this dataset.**

To simplify the problem, we will resize all images as 256×256. You can use the codes ["2-1. pre-processing image (train & val).R"](https://github.com/xup6fup/MxNetR-YOLO/blob/master/code/1.%20Processing%20data/2-1.%20pre-processing%20image%20(train%20%26%20val).R) and ["2-2. pre-processing image (test).R"](https://github.com/xup6fup/MxNetR-YOLO/blob/master/code/1.%20Processing%20data/2-2.%20pre-processing%20image%20(test).R) to do this work. Resized images will be converted as .RData and they totally used about 15GB for storing them.

After we get the bounding box infomation in pevious stage, we can caculate the anchor boxes by k-mean clustering analysis. In yolo v3, there are 9 anchor boxes belonging feature map with 8 stride (3 smallest), 16 stride (3 meddle size), and 32 stride (3 biggest), respectively. You can use the codes ["3. Define the anchor boxes (for yolo v3).R"](https://github.com/xup6fup/MxNetR-YOLO/blob/master/code/1.%20Processing%20data/3.%20Define%20the%20anchor%20boxes%20(for%20yolo%20v3).R) for conducting this process. Finally, we will get the **anchor_boxs (yolo v3).RData** for further application.

2. Training stage
---

The first step for using MxNet to train a yolo model is to build an iterator. You can use the codes ["1. Encode, Decode & Iterator.R"](https://github.com/xup6fup/MxNetR-YOLO/blob/master/code/2.%20Training/1.%20Encode%2C%20Decode%20%26%20Iterator.R) for conducting this process. It is worth noting that bounding boxes are needed to encode as a special form for following training. Moreover, the encoded labels also need to pass a decoding process for restoring bounding boxes. The encode and decode function are the core of the yolo model. If you want to clearly understand the principle of yolo model, you can dismantle these functions to learn. The test codes for generating images are also included in that code, let's try it!

The next step is to define the model architecture. We use a pretrained model (training by imagenet for image recognition) and fine tune it. Here I select the resnet-34 to train a yolo model. The resnet-34 model can be downloaded from [MxNet model zoo](http://data.mxnet.io/models/imagenet/). The model should be saved in folder 'model/pretrained model' for following use. Of course, you can also select other model. The code ["2. Model architecture.R"](https://github.com/xup6fup/MxNetR-YOLO/blob/master/code/2.%20Training/2.%20Model%20architecture.R) includes yolo predict architecture and loss function, you can try to learn yolo v3 from these codes.

Now we can start to train this model! Because yolo v2 suggest that multi-scale training, so the training code is complex. The support functions can be found from ["3. Support functions.R"](https://github.com/xup6fup/MxNetR-YOLO/blob/master/code/2.%20Training/3.%20Support%20functions.R), and finally you can use ["4. Train a yolo model.R"](https://github.com/xup6fup/MxNetR-YOLO/blob/master/code/2.%20Training/4.%20Train%20a%20yolo%20model.R) for training this model. It is worth noting that the total training time in this sample is about 35 hours in single P100 GPU server.

3. Model performance
---

Finally, we get a 50.4% MAP50 in testing set. Following image is the selected predicting results by our model:

<p align="center">
  <img src="Pred_example.jpeg">
</p>

You can use the code ["5. Test the model performance.R"](https://github.com/xup6fup/MxNetR-YOLO/blob/master/code/2.%20Training/5.%20Test%20the%20model%20performance.R) for conducting this process. Because this is a simple example for yolo v3, our database only includes 4,008 training images and 1,003 validation images, so I consider this result is very good.
