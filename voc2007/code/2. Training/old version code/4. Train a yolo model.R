
source('voc2007/code/2. Training/1. Encode, Decode & Iterator.R')
source('voc2007/code/2. Training/2. Model architecture.R')
source('voc2007/code/2. Training/3. Support functions.R')

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

for (iter in 1:20) {
  
  message('start to ', iter, ' iter')
  
  if (iter == 1) {
    ARG.PARAMS <- new_arg$arg.params
  } else {
    ARG.PARAMS <- YOLO_model$arg.params
  }
  
  if (iter %% 5 == 1) {
    dynamic_img_size <- 256
  } else if (iter %% 5 == 2) {
    dynamic_img_size <- 288
  } else if (iter %% 5 == 3) {
    dynamic_img_size <- 192
  } else if (iter %% 5 == 4) {
    dynamic_img_size <- 320
  } else if (iter %% 5 == 0) {
    dynamic_img_size <- 224
  }
  
  my_logger <- mx.metric.logger$new()
  
  my_iter <- my_iterator_func(iter = NULL, batch_size = 16, img_size = dynamic_img_size, resize_method = 'nearest',
                              sample = 'train', aug_crop = TRUE, aug_flip = TRUE)
  
  YOLO_model <- mx.model.FeedForward.create(final_yolo_loss, X = my_iter,
                                            ctx = mx.gpu(), begin.round = 1, num.round = 50,
                                            optimizer = 'adam', learning.rate = 1e-3/iter, beta1 = 0.9, beta2 = 0.999, wd = 1e-4,
                                            fixed.param = Layer_to_fixed, arg.params = ARG.PARAMS, 
                                            eval.metric = my.eval.metric.loss,
                                            input.names = 'data', output.names = c('label1', 'label2', 'label3'),
                                            batch.end.callback = my.callback_batch(batch.size = 16, frequency = 50),
                                            epoch.end.callback = my.callback_epoch(out_symbol = final_yolo_loss, logger = my_logger,
                                                                                   prefix = 'model/yolo model/yolo_v3',
                                                                                   fixed.params = ARG.PARAMS[names(ARG.PARAMS) %in% Layer_to_fixed],
                                                                                   period = 1))
  
  deleted_list <- list.files('model/yolo model', pattern = '*.params', full.names = TRUE)
  deleted_list <- deleted_list[!grepl('0000', deleted_list, fixed = TRUE)]
  mx.model.save(YOLO_model, 'model/yolo model/yolo_v3', 0)
  file.remove(deleted_list)
  
}
