
# Libraries

library(mxnet)
library(magrittr)

## Define the model architecture
## Use pre-trained model and fine tuning

# Load Mobile Net V2

Pre_Trained_model <- mx.model.load('model/pretrained model/mobilev2', 0)

# Get the internal output

Mobile_V2_symbol <- Pre_Trained_model$symbol

Mobile_V2_All_layer <- Mobile_V2_symbol$get.internals()

lvl1_out <- which(Mobile_V2_All_layer$outputs == 'conv3_2_linear_bn_output') %>% Mobile_V2_All_layer$get.output()
lvl2_out <- which(Mobile_V2_All_layer$outputs == 'conv4_7_linear_bn_output') %>% Mobile_V2_All_layer$get.output()
lvl3_out <- which(Mobile_V2_All_layer$outputs == 'conv5_3_linear_bn_output') %>% Mobile_V2_All_layer$get.output()

# mx.symbol.infer.shape(lvl3_out, data = c(256, 256, 3, 7))$out.shapes

# conv3_2_linear_bn_output out shape = 32 32 32 n (if input shape = 256 256 3 n)
# conv4_7_linear_bn_output out shape = 16 16 96 n (if input shape = 256 256 3 n)
# conv5_3_linear_bn_output out shape = 8 8 160 n (if input shape = 256 256 3 n)

# Load Resnet-101

Pre_Trained_model <- mx.model.load('model/pretrained model/resnet-101', 0)

# Get the internal output

Resnet_symbol <- Pre_Trained_model$symbol

Resnet_All_layer <- Resnet_symbol$get.internals()

lvl1_out <- which(Resnet_All_layer$outputs == '_plus6_output') %>% Resnet_All_layer$get.output()
lvl2_out <- which(Resnet_All_layer$outputs == '_plus29_output') %>% Resnet_All_layer$get.output()
lvl3_out <- which(Resnet_All_layer$outputs == 'relu1_output') %>% Resnet_All_layer$get.output()

# mx.symbol.infer.shape(lvl3_out, data = c(256, 256, 3, 7))$out.shapes

# _plus6_output out shape = 32 32 512 n (if input shape = 256 256 3 n)
# _plus29_output out shape = 16 16 1024 n (if input shape = 256 256 3 n)
# relu1_output out shape = 8 8 2048 n (if input shape = 256 256 3 n)

# Load Resnet-34

Pre_Trained_model <- mx.model.load('model/pretrained model/resnet-34', 0)

# Get the internal output

Resnet_symbol <- Pre_Trained_model$symbol

Resnet_All_layer <- Resnet_symbol$get.internals()

lvl1_out <- which(Resnet_All_layer$outputs == '_plus6_output') %>% Resnet_All_layer$get.output()
lvl2_out <- which(Resnet_All_layer$outputs == '_plus12_output') %>% Resnet_All_layer$get.output()
lvl3_out <- which(Resnet_All_layer$outputs == 'relu1_output') %>% Resnet_All_layer$get.output()

# mx.symbol.infer.shape(lvl3_out, data = c(256, 256, 3, 7))$out.shapes

# _plus6_output out shape = 32 32 128 n (if input shape = 256 256 3 n)
# _plus12_output out shape = 16 16 256 n (if input shape = 256 256 3 n)
# relu1_output out shape = 8 8 512 n (if input shape = 256 256 3 n)

# Convolution layer for specific mission and training new parameters



