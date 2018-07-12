
# Libraries

library(mxnet)
library(magrittr)

## Define the model architecture
## Use pre-trained model and fine tuning

# Load Mobile Net V2

Pre_Trained_model <- mx.model.load('Model/Pre-trained model/mobilev2', 0)

Mobile_V2_symbol <- Pre_Trained_model$symbol

Mobile_V2_All_layer <- Mobile_V2_symbol$get.internals()


