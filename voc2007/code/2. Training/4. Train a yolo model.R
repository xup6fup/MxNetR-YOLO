
source('voc2007/code/2. Training/1. Encode, Decode & Iterator.R')
source('voc2007/code/2. Training/2. Model architecture.R')
source('voc2007/code/2. Training/3. Support functions.R')

## initiate Parameter for model


new_arg <- mxnet:::mx.model.init.params(symbol = final_yolo_loss, 
                                        input.shape = list(data = c(224, 224, 3, 13), 
                                                           label1 = c(28, 28, 75, 13), 
                                                           label2 = c(14, 14, 75, 13), 
                                                           label3 = c(7, 7, 75, 13)), 
                                        output.shape = NULL, initializer = mx.init.uniform(0.01), 
                                        ctx = CTX)

message('The total number of parameters = ', sum(sapply(lapply(new_arg$arg.params, dim), prod)))

## Bind Pre-trained Parameter into model

Pre_trained_ARG <- Pre_Trained_model$arg.params

ARG_in_net_name <- names(Pre_trained_ARG) %>% .[. %in% names(new_arg$arg.params)]  # remove paramter does not in model

message('number of pre trained paramters: ', length(ARG_in_net_name))

for (i in 1:length(ARG_in_net_name)){
  new_arg$arg.params[names(new_arg$arg.params) == ARG_in_net_name[i]] <- Pre_trained_ARG[names(Pre_trained_ARG) == ARG_in_net_name[i]]
}

## Define fixed layer

Layer_to_fixed <- ARG_in_net_name

# Model Training

my_iter_list <- list()

for (k in c(288, 192, 320, 224, 256)) {
  
  my_iter_list[[length(my_iter_list)+1]] <- my_iterator_func(iter = NULL, batch_size = 16,
                                                             img_size = k, resize_method = 'nearest',
                                                             sample = 'train', aug_crop = TRUE, aug_flip = TRUE)
  
}

val_iter <- my_iterator_func(iter = NULL, batch_size = 50,
                             img_size = 256, resize_method = 'nearest',
                             sample = 'val', aug_crop = FALSE, aug_flip = FALSE)

YOLO_model <- my.yolo_trainer(symbol = final_yolo_loss, Iterator_list = my_iter_list, val_iter = val_iter,
                              ctx = mx.gpu(), num_round = 10, num_iter = 10,
                              start_val = 1, start_unfixed = 2,
                              prefix = 'model/yolo model (voc2007)/yolo_v3 (1)',
                              Fixed_NAMES = Layer_to_fixed, ARG.PARAMS = new_arg$arg.params)
