
# Libraries

library(mxnet)
library(magrittr)

## Define the model architecture
## Use pre-trained model and fine tuning

# Load Mobile Net V2

Pre_Trained_model <- mx.model.load('model/pretrained model/mobilev2', 0)

# Get the internal output
# conv3_2_linear_bn_output out shape = 32 32 32 n (if input shape = 256 256 3 n)
# conv4_7_linear_bn_output out shape = 16 16 96 n (if input shape = 256 256 3 n)
# conv5_3_linear_bn_output out shape = 8 8 160 n (if input shape = 256 256 3 n)

Mobile_V2_symbol <- Pre_Trained_model$symbol

Mobile_V2_All_layer <- Mobile_V2_symbol$get.internals()

lvl1_out <- which(Mobile_V2_All_layer$outputs == 'conv3_2_linear_bn_output') %>% Mobile_V2_All_layer$get.output()
lvl2_out <- which(Mobile_V2_All_layer$outputs == 'conv4_7_linear_bn_output') %>% Mobile_V2_All_layer$get.output()
lvl3_out <- which(Mobile_V2_All_layer$outputs == 'conv5_3_linear_bn_output') %>% Mobile_V2_All_layer$get.output()

# Convolution layer for specific mission and training new parameters



