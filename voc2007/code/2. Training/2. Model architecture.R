
# Libraries

library(mxnet)
library(magrittr)

## Define the model architecture
## Use pre-trained model and fine tuning

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

# 1. Additional 2 layers architecture for better learning
 
upsampling_function <- function (updata, downdata, num_filters = 256, name = 'lvl1') {
   
  bn <- mx.symbol.BatchNorm(data = updata, fix_gamma = FALSE, name = paste0(name, '_up_bn'))
  relu <- mx.symbol.Activation(data = bn, act.type = "relu", name = paste0(name, '_up_relu'))
  deconv <- mx.symbol.Deconvolution(data = relu, kernel = c(2, 2), stride = c(2, 2), pad = c(0, 0),
                                    no.bias = TRUE, num.filter = num_filters, name = paste0(name, '_up_deconv'))
  new_list <- list()
  new_list[[1]] <- downdata
  new_list[[2]] <- deconv
  
  concat_map <- mx.symbol.concat(data = new_list, num.args = 2, dim = 1, name = paste0(name, "_concat_map"))
   
  return(concat_map)
   
}
 
YOLO_map_function <- function (indata, num_filters = c(128, 128), final_map = 75, num_box = 3, name = 'lvl1') {
  
  for (i in 1:length(num_filters)) {
    
    if (i == 1) {
      bn <- mx.symbol.BatchNorm(data = indata, fix_gamma = FALSE, name = paste0(name, '_bn', i))
    } else {
      bn <- mx.symbol.BatchNorm(data = conv, fix_gamma = FALSE, name = paste0(name, '_bn', i))
    }
    
    relu <- mx.symbol.Activation(data = bn, act.type = "relu", name = paste0(name, '_relu', i))
    
    conv <- mx.symbol.Convolution(data = relu, kernel = c(3, 3), stride = c(1, 1), pad = c(1, 1),
                                  no.bias = TRUE, num.filter = num_filters[i], name = paste0(name, '_conv', i))
    
  }
  
  bn <- mx.symbol.BatchNorm(data = conv, fix_gamma = FALSE, name = paste0(name, '_bn.final'))
  relu <- mx.symbol.Activation(data = bn, act.type = "relu", name = paste0(name, '_bn.final'))
  conv <- mx.symbol.Convolution(data = relu, kernel = c(1, 1), stride = c(1, 1), pad = c(0, 0),
                                no.bias = FALSE, num.filter = final_map, name = paste0(name, '_linearmap'))  
  
  inter_split <- mx.symbol.SliceChannel(data = conv, num_outputs = final_map,
                                        axis = 1, squeeze_axis = FALSE, name = paste0(name, "_inter_split"))
  
  new_list <- list()
  
  for (k in 1:final_map) {
    if (!(final_map %% num_box) %in% c(2:5)) {
      new_list[[k]] <- mx.symbol.Activation(inter_split[[k]], act.type = 'sigmoid', name = paste0(name, "_yolomap_", k))
    }
  }
  
  yolomap <- mx.symbol.concat(data = new_list, num.args = final_map, dim = 1, name = paste0(name, "_yolomap"))
  
  return(yolomap)
  
}

up_lvl2 <- upsampling_function(updata = lvl3_out, downdata = lvl2_out, num_filters = 256, name = 'lvl2')
up_lvl1 <- upsampling_function(updata = up_lvl2, downdata = lvl1_out, num_filters = 256, name = 'lvl2')

yolomap_list <- list()

yolomap_list[[1]] <- YOLO_map_function(indata = up_lvl1, num_filters = c(512, 512), final_map = 75, name = 'lvl1')
yolomap_list[[2]] <- YOLO_map_function(indata = up_lvl2, num_filters = c(512, 512), final_map = 75, name = 'lvl2')
yolomap_list[[3]] <- YOLO_map_function(indata = lvl3_out, num_filters = c(512, 512), final_map = 75, name = 'lvl3')

# 2. Custom loss function

MSE_loss_function <- function (indata, inlabel, obj, lambda, pre_sqrt = FALSE) {
  
  if (pre_sqrt) {
    indata <- mx.symbol.sqrt(indata)
    inlabel <- mx.symbol.sqrt(inlabel)
  }
  
  diff_pred_label <- mx.symbol.broadcast_minus(lhs = indata, rhs = inlabel)
  square_diff_pred_label <- mx.symbol.square(data = diff_pred_label)
  obj_square_diff_loss <- mx.symbol.broadcast_mul(lhs = obj, rhs = square_diff_pred_label)
  MSE_loss <- mx.symbol.mean(data = obj_square_diff_loss, axis = 0:3, keepdims = FALSE)
  
  return(MSE_loss)
  
}

CE_loss_function <- function (indata, inlabel, obj, lambda, eps = 1e-4) {
  
  log_pred_1 <- mx.symbol.log(data = indata + eps)
  log_pred_2 <- mx.symbol.log(data = 1 - indata + eps)
  multiple_log_pred_label_1 <- mx.symbol.broadcast_mul(lhs = log_pred_1, rhs = inlabel)
  multiple_log_pred_label_2 <- mx.symbol.broadcast_mul(lhs = log_pred_2, rhs = 1 - inlabel)
  obj_weighted_loss <- mx.symbol.broadcast_mul(lhs = obj, rhs = multiple_log_pred_label_1 + multiple_log_pred_label_2)
  average_CE_loss <- mx.symbol.mean(data = obj_weighted_loss, axis = 0:3, keepdims = FALSE)
  CE_loss <- 0 - average_CE_loss * lambda
  
  return(CE_loss)
  
}

YOLO_loss_function <- function (indata, inlabel, final_map = 75, num_box = 3, lambda = 1000, name = 'lvl1') {
  
  num_feature <- final_map/num_box
  
  my_loss <- 0
  
  yolomap_split <- mx.symbol.SliceChannel(data = indata, num_outputs = final_map, 
                                          axis = 1, squeeze_axis = FALSE, name = paste(name, '_yolomap_split'))
  
  label_split <- mx.symbol.SliceChannel(data = inlabel, num_outputs = final_map, 
                                        axis = 1, squeeze_axis = FALSE, name = paste(name, '_label_split'))
  
  for (j in 1:num_box) {
    for (k in 1:num_feature) {
      if (k %in% 1:5) {weight <- 1} else {weight <- 0.2}
      if (!k %in% c(2:5)) {
        my_loss <- my_loss + CE_loss_function(indata = yolomap_split[[(j-1)*num_feature+k]],
                                              inlabel = label_split[[(j-1)*num_feature+k]],
                                              obj = label_split[[(j-1)*num_feature+1]],
                                              lambda = lambda * weight,
                                              eps = 1e-4)
        if (k == 1) {
          my_loss <- my_loss + CE_loss_function(indata = yolomap_split[[(j-1)*num_feature+k]],
                                                inlabel = label_split[[(j-1)*num_feature+k]],
                                                obj = 1 - label_split[[(j-1)*num_feature+1]],
                                                lambda = 1,
                                                eps = 1e-4)
        }
      } else {
        my_loss <- my_loss + MSE_loss_function(indata = yolomap_split[[(j-1)*num_feature+k]],
                                               inlabel = label_split[[(j-1)*num_feature+k]],
                                               obj = label_split[[(j-1)*num_feature+1]],
                                               lambda = lambda * weight, pre_sqrt = FALSE)
      }
    }
  }
  
  return(my_loss)
  
}

label1 <- mx.symbol.Variable(name = "label1")
label2 <- mx.symbol.Variable(name = "label2")
label3 <- mx.symbol.Variable(name = "label3")

lvl1_loss <- YOLO_loss_function(indata = yolomap_list[[1]], inlabel = label1, final_map = 75, num_box = 3, lambda = 10, name = 'lvl1')
lvl2_loss <- YOLO_loss_function(indata = yolomap_list[[2]], inlabel = label2, final_map = 75, num_box = 3, lambda = 10, name = 'lvl2')
lvl3_loss <- YOLO_loss_function(indata = yolomap_list[[3]], inlabel = label3, final_map = 75, num_box = 3, lambda = 10, name = 'lvl3')

final_yolo_loss <- mx.symbol.MakeLoss(data = lvl1_loss + lvl2_loss + lvl3_loss)
