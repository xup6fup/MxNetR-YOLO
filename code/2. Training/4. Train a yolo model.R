
source('code/2. Training/1. Encode, Decode & Iterator.R')
source('code/2. Training/2. Model architecture.R')
source('code/2. Training/3. Support functions.R')

## initiate Parameter for model

new_arg <- mxnet:::mx.model.init.params(symbol = final_yolo_loss, 
                                        input.shape = list(data = c(224, 224, 3, 13), 
                                                           label1 = c(28, 28, 75, 13), 
                                                           label2 = c(14, 14, 75, 13), 
                                                           label3 = c(7, 7, 75, 13)), 
                                        output.shape = NULL, initializer = mxnet:::mx.init.Xavier(rnd_type = "uniform", magnitude = 2.24), 
                                        ctx = CTX)

## Bind Pre-trained Parameter into model

Pre_trained_ARG <- Pre_Trained_model$arg.params

ARG_in_net_name <- names(Pre_trained_ARG) %>% .[. %in% names(new_arg$arg.params)]  # remove paramter does not in model

for (i in 1:length(ARG_in_net_name)){
  new_arg$arg.params[names(new_arg$arg.params) == ARG_in_net_name[i]] <- Pre_trained_ARG[names(Pre_trained_ARG) == ARG_in_net_name[i]]
}

## Define fixed layer

Layer_to_fixed <- ARG_in_net_name

# Model Training

my_iter_list <- list()

for (k in c(256, 288, 192, 320, 224)) {
  
  my_iter_list[[length(my_iter_list)+1]] <- my_iterator_func(iter = NULL, batch_size = 16,
                                                             img_size = k, resize_method = 'nearest',
                                                             sample = 'train', aug_crop = TRUE, aug_flip = TRUE)
  
}

val_iter <- my_iterator_func(iter = NULL, batch_size = 50,
                             img_size = 256, resize_method = 'nearest',
                             sample = 'val', aug_crop = FALSE, aug_flip = FALSE)

YOLO_model <- my.yolo_trainer(symbol = final_yolo_loss, Iterator_list = my_iter_list, val_iter = val_iter,
                              ctx = mx.gpu(), num_round = 5, num_iter = 30,
                              prefix = 'model/yolo model/yolo_v3',
                              Fixed_NAMES = Layer_to_fixed, ARG.PARAMS = new_arg$arg.params)
