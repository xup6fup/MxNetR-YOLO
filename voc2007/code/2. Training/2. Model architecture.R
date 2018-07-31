
# Libraries

library(mxnet)
library(magrittr)

## Define the model architecture
## Use pre-trained model and fine tuning

# Load MobileNet v2

#Pre_Trained_model <- mx.model.load('model/pretrained model/mobilev2', 0)
Pre_Trained_model <- mx.model.load('model/yolo model (voc2007)/yolo_v3 (1)', 0)

# Get the internal output

Mobile_symbol <- Pre_Trained_model$symbol

Mobile_All_layer <- Mobile_symbol$get.internals()

lvl1_out <- which(Mobile_All_layer$outputs == 'block_4_6_output') %>% Mobile_All_layer$get.output()
lvl2_out <- which(Mobile_All_layer$outputs == 'block_5_2_output') %>% Mobile_All_layer$get.output()
lvl3_out <- which(Mobile_All_layer$outputs == 'conv6_3_linear_bn_output') %>% Mobile_All_layer$get.output()

# mx.symbol.infer.shape(lvl3_out, data = c(256, 256, 3, 7))$out.shapes

# conv3_2_linear_bn_output out shape = 32 32 32 n (if input shape = 256 256 3 n)
# conv4_7_linear_bn_output out shape = 16 16 96 n (if input shape = 256 256 3 n)
# conv6_3_linear_bn_output out shape = 8 8 320 n (if input shape = 256 256 3 n)

# Convolution layer for specific mission and training new parameters

# 1. Additional some architecture for better learning

DECONV_function <- function (updata, downdata, num_filters = 256, name = 'lvl1') {
  
  deconv <- mx.symbol.Deconvolution(data = updata, kernel = c(2, 2), stride = c(2, 2), pad = c(0, 0),
                                    no.bias = TRUE, num.filter = num_filters,
                                    name = paste0(name, '_deconv'))
  deconv_bn <- mx.symbol.BatchNorm(data = deconv, fix_gamma = FALSE, name = paste0(name, '_deconv_bn'))
  deconv_relu <- mx.symbol.LeakyReLU(data = deconv_bn, act.type = 'leaky', slope = 0.1, name = paste0(name, '_deconv_relu'))
  
  new_list <- list()
  new_list[[1]] <- downdata
  new_list[[2]] <- deconv_relu
  
  concat_map <- mx.symbol.concat(data = new_list, num.args = 2, dim = 1, name = paste0(name, "_concat_map"))
  
  return(concat_map)
  
}

DWCONV_function <- function (indata, num_filters = 256, Inverse_coef = 6, residual = TRUE, name = 'lvl1', stage = 1) {
  
  expend_conv <- mx.symbol.Convolution(data = indata, kernel = c(1, 1), stride = c(1, 1), pad = c(0, 0),
                                       no.bias = TRUE, num.filter = num_filters * Inverse_coef,
                                       name = paste0(name, '_', stage, '_expend'))
  expend_bn <- mx.symbol.BatchNorm(data = expend_conv, fix_gamma = FALSE, name = paste0(name, '_', stage, '_expend_bn'))
  expend_relu <- mx.symbol.LeakyReLU(data = expend_bn, act.type = 'leaky', slope = 0.1, name = paste0(name, '_', stage, '_expend_relu'))
  
  dwise_conv <- mx.symbol.Convolution(data = expend_relu, kernel = c(3, 3), stride = c(1, 1), pad = c(1, 1),
                                      no.bias = TRUE, num.filter = num_filters * Inverse_coef, num.group = num_filters * Inverse_coef,
                                      name = paste0(name, '_', stage, '_dwise'))
  dwise_bn <- mx.symbol.BatchNorm(data = dwise_conv, fix_gamma = FALSE, name = paste0(name, '_', stage, '_dwise_bn'))
  dwise_relu <- mx.symbol.LeakyReLU(data = dwise_bn, act.type = 'leaky', slope = 0.1, name = paste0(name, '_', stage, '_dwise_relu'))
  
  restore_conv <- mx.symbol.Convolution(data = dwise_relu, kernel = c(1, 1), stride = c(1, 1), pad = c(0, 0),
                                        no.bias = TRUE, num.filter = num_filters,
                                        name = paste0(name, '_', stage, '_restore'))
  restore_bn <- mx.symbol.BatchNorm(data = restore_conv, fix_gamma = FALSE, name = paste0(name, '_', stage, '_restore_bn'))
  
  if (residual) {
    
    block <- mx.symbol.broadcast_plus(lhs = indata, rhs = restore_bn, name = paste0(name, '_', stage, '_block'))
    return(block)
    
  } else {
    
    restore_relu <- mx.symbol.LeakyReLU(data = restore_bn, act.type = 'leaky', slope = 0.1, name = paste0(name, '_', stage, '_restore_relu'))
    return(restore_relu)
    
  }
  
  
  
}

CONV_function <- function (indata, num_filters = 256, name = 'lvl1', stage = 1) {

  conv <- mx.symbol.Convolution(data = indata, kernel = c(3, 3), stride = c(1, 1), pad = c(1, 1),
                                no.bias = TRUE, num.filter = num_filters,
                                name = paste0(name, '_', stage, '_conv'))
  bn <- mx.symbol.BatchNorm(data = conv, fix_gamma = FALSE, name = paste0(name, '_', stage, '_bn'))
  relu <- mx.symbol.LeakyReLU(data = bn, act.type = 'leaky', slope = 0.1, name = paste0(name, '_', stage, '_relu'))

  return(relu)
  
}

YOLO_map_function <- function (indata, final_map = 75, num_box = 3, drop = 0.2, name = 'lvl1') {
  
  dp <- mx.symbol.Dropout(data = indata, p = drop, name = paste0(name, '_drop'))
  
  conv <- mx.symbol.Convolution(data = dp, kernel = c(1, 1), stride = c(1, 1), pad = c(0, 0),
                                no.bias = FALSE, num.filter = final_map, name = paste0(name, '_linearmap'))
  
  inter_split <- mx.symbol.SliceChannel(data = conv, num_outputs = final_map,
                                        axis = 1, squeeze_axis = FALSE, name = paste0(name, "_inter_split"))
  
  new_list <- list()
  
  for (k in 1:final_map) {
    if (!(k %% num_box) %in% c(4:5)) {
      new_list[[k]] <- mx.symbol.Activation(inter_split[[k]], act.type = 'sigmoid', name = paste0(name, "_yolomap_", k))
    } else {
      new_list[[k]] <- inter_split[[k]]
    }
  }
  
  yolomap <- mx.symbol.concat(data = new_list, num.args = final_map, dim = 1, name = paste0(name, "_yolomap"))
  
  return(yolomap)
  
}

lvl3_conv_1 <- DWCONV_function(indata = lvl3_out, num_filters = 320, Inverse_coef = 3, residual = TRUE, name = 'lvl3', stage = 1)
lvl3_conv_2 <- DWCONV_function(indata = lvl3_conv_1, num_filters = 320, Inverse_coef = 3, residual = TRUE, name = 'lvl3', stage = 2)
lvl3_conv_3 <- CONV_function(indata = lvl3_conv_2, num_filters = 320, name = 'lvl3', stage = 3)

lvl2_cat <- DECONV_function(updata = lvl3_conv_3, downdata = lvl2_out, num_filters = 160, name = 'lvl2')

lvl2_conv_1 <- DWCONV_function(indata = lvl2_cat, num_filters = 256, Inverse_coef = 3, residual = TRUE, name = 'lvl2', stage = 1)
lvl2_conv_2 <- DWCONV_function(indata = lvl2_conv_1, num_filters = 256, Inverse_coef = 3, residual = TRUE, name = 'lvl2', stage = 2)
lvl2_conv_3 <- CONV_function(indata = lvl2_conv_2, num_filters = 256, name = 'lvl2', stage = 3)

lvl1_cat <- DECONV_function(updata = lvl2_conv_3, downdata = lvl1_out, num_filters = 128, name = 'lvl1')

lvl1_conv_1 <- DWCONV_function(indata = lvl1_cat, num_filters = 192, Inverse_coef = 3, residual = TRUE, name = 'lvl1', stage = 1)
lvl1_conv_2 <- DWCONV_function(indata = lvl1_conv_1, num_filters = 192, Inverse_coef = 3, residual = TRUE, name = 'lvl1', stage = 2)
lvl1_conv_3 <- CONV_function(indata = lvl1_conv_2, num_filters = 192, name = 'lvl1', stage = 3)

yolomap_list <- list()

yolomap_list[[1]] <- YOLO_map_function(indata = lvl1_conv_3, final_map = 75, drop = 0.2, name = 'lvl1')
yolomap_list[[2]] <- YOLO_map_function(indata = lvl2_conv_3, final_map = 75, drop = 0.2, name = 'lvl2')
yolomap_list[[3]] <- YOLO_map_function(indata = lvl3_conv_3, final_map = 75, drop = 0.2, name = 'lvl3')

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
