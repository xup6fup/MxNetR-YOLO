
source('pikachu/code/2. Training/1. Encode, Decode & Iterator.R')
source('pikachu/code/2. Training/2. Model architecture.R')
source('pikachu/code/2. Training/3. Support functions.R')

## initiate Parameter for model

new_arg <- mxnet:::mx.model.init.params(symbol = final_yolo_loss, 
                                        input.shape = list(data = c(224, 224, 3, 13), 
                                                           label = c(7, 7, 6, 13)), 
                                        output.shape = NULL, initializer = mxnet:::mx.init.Xavier(rnd_type = "uniform", magnitude = 2.24), 
                                        ctx = CTX)

## Bind Pre-trained Parameter into model

Pre_trained_ARG <- Pre_Trained_model$arg.params

ARG_in_net_name <- names(Pre_trained_ARG) %>% .[. %in% names(new_arg$arg.params)]  # remove paramter does not in model

for (i in 1:length(ARG_in_net_name)){
  new_arg$arg.params[names(new_arg$arg.params) == ARG_in_net_name[i]] <- Pre_trained_ARG[names(Pre_trained_ARG) == ARG_in_net_name[i]]
}

ARG.PARAMS <- new_arg$arg.params

# Model Training

my_logger <- mx.metric.logger$new()

my_iter <- my_iterator_func(iter = NULL, batch_size = 16, img_size = 256, aug_crop = TRUE, aug_flip = TRUE)

YOLO_model <- mx.model.FeedForward.create(final_yolo_loss, X = my_iter,
                                          ctx = mx.gpu(), begin.round = 1, num.round = 100,
                                          optimizer = 'sgd', learning.rate = 1e-2, momentum = 0.9, wd = 1e-4,
                                          arg.params = ARG.PARAMS, 
                                          eval.metric = my.eval.metric.loss,
                                          input.names = 'data', output.names = 'label',
                                          batch.end.callback = my.callback_batch(batch.size = 16, frequency = 10),
                                          epoch.end.callback = my.callback_epoch(out_symbol = yolomap, logger = my_logger,
                                                                                 prefix = 'model/yolo model (pikachu)/yolo_v1',
                                                                                 period = 1))

