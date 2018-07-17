
# Libraries

library(mxnet)
library(data.table)
library(magrittr)

# Custom callback function

my.eval.metric.loss <- mx.metric.custom(
  name = "multi_part_loss",
  function(label, pred) {
    return(as.array(pred))
  }
)

my.callback_batch <- function (batch.size = 16, frequency = 10) {
  function(iteration, nbatch, env, verbose = TRUE) {
    count <- nbatch
    if (is.null(env$count)) 
      env$count <- 0
    if (is.null(env$init)) 
      env$init <- FALSE
    if (env$count > count) 
      env$init <- FALSE
    env$count = count
    if (env$init) {
      if (count%%frequency == 0 && !is.null(env$metric)) {
        time <- as.double(difftime(Sys.time(), env$tic, 
                                   units = "secs"))
        speed <- frequency * batch.size/time
        result <- env$metric$get(env$train.metric)
        if (nbatch != 0 & verbose) {
          message(paste0("Batch [", nbatch, "] Speed: ", 
                         formatC(speed, 3, format = "f"), " samples/sec Train-", result$name, 
                         "=", as.array(result$value)))
        }
        env$tic = Sys.time()
      }
    }
    else {
      env$init <- TRUE
      env$tic <- Sys.time()
    }
  }
}


my.callback_epoch <- function (out_symbol, logger = NULL, 
                               prefix = 'model/yolo model/yolo_v1',
                               fixed.params = NULL,
                               period = 1) {
  function(iteration, nbatch, env, verbose = TRUE) {
    if (iteration%%period == 0) {
      env_model <- env$model
      env_all_layers <- env_model$symbol$get.internals()
      model_write_out <- list(symbol = out_symbol,
                              arg.params = env_model$arg.params,
                              aux.params = env_model$aux.params)
      model_write_out[[2]] <- append(model_write_out[[2]], fixed.params)
      class(model_write_out) <- "MXFeedForwardModel"
      mx.model.save(model_write_out, prefix, iteration)
      if (verbose) {
        message(sprintf("Model checkpoint saved to %s-%04d.params", prefix, iteration))
      }
    }
    if (!is.null(logger)) {
      if (class(logger) != "mx.metric.logger") {
        stop("Invalid mx.metric.logger.")
      } else {
        result <- env$metric$get(env$train.metric)
        logger$train <- c(logger$train, result$value)
        if (!is.null(env$eval.metric)) {
          result <- env$metric$get(env$eval.metric)
          logger$eval <- c(logger$eval, result$value)
        }
      }
    }
    return(TRUE)
  }
}

my_predict <- function (model, img, ctx = mx.gpu()) {
  
  require(magrittr)
  
  all_layers <- model$symbol$get.internals()
  
  lvl1_output <- which(all_layers$outputs == 'lvl1_yolomap_output') %>% all_layers$get.output()
  lvl2_output <- which(all_layers$outputs == 'lvl2_yolomap_output') %>% all_layers$get.output()
  lvl3_output <- which(all_layers$outputs == 'lvl3_yolomap_output') %>% all_layers$get.output()
  
  out <- mx.symbol.Group(c(lvl1_output, lvl2_output, lvl3_output))
  executor <- mx.simple.bind(symbol = out, data = dim(img), ctx = ctx)
  
  need_arg <- ls(mx.symbol.infer.shape(out, data = c(224, 224, 3, 7))$arg.shapes)
  
  mx.exec.update.arg.arrays(executor, model$arg.params[names(model$arg.params) %in% need_arg], match.name = TRUE)
  mx.exec.update.aux.arrays(executor, model$aux.params, match.name = TRUE)
  if (class(img)!='MXNDArray') {img <- mx.nd.array(img)}
  mx.exec.update.arg.arrays(executor, list(data = img), match.name = TRUE)
  mx.exec.forward(executor, is.train = FALSE)
  
  pred_list <- list()
  
  pred_list[[1]] <- as.array(executor$ref.outputs$lvl1_yolomap_output)
  pred_list[[2]] <- as.array(executor$ref.outputs$lvl2_yolomap_output)
  pred_list[[3]] <- as.array(executor$ref.outputs$lvl3_yolomap_output)
  
  return(pred_list)
  
}

IoU_function <- function (label, pred) {
  
  overlap_width <- min(label[,2], pred[,2]) - max(label[,1], pred[,1])
  overlap_height <- min(label[,3], pred[,3]) - max(label[,4], pred[,4])
  
  if (overlap_width > 0 & overlap_height > 0) {
    
    pred_size <- (pred[,2]-pred[,1])*(pred[,3]-pred[,4])
    label_size <- (label[,2]-label[,1])*(label[,3]-label[,4])
    overlap_size <- overlap_width * overlap_height
    
    return(overlap_size/(pred_size + label_size - overlap_size))
    
  } else {
    
    return(0)
    
  }
  
}

AP_function <- function (vbYreal, vdYhat) {
  
  vbYreal_sort_d <- vbYreal[order(vdYhat, decreasing=TRUE)]
  P_list <- cumsum(vbYreal_sort_d) * vbYreal_sort_d / seq_along(vbYreal_sort_d)
  P_list <- P_list[P_list!=0]
  diff_P_list <- diff(P_list)
  diff_P_list[diff_P_list < 0] <- 0
  P_list <- P_list + c(diff_P_list, 0)
  return(mean(P_list))
  
}

model_AP_func <- function (model, Iterator, IoU_cut = 0.5) {
  
  Iterator$reset()
  
  label_box_info <- list()
  pred_box_info <- list()
  
  num_batch <- 1
  
  while (Iterator$iter.next()) {
    
    vlist <- Iterator$value()
    img_array <- vlist$data
    
    label_list <- vlist[2:4]
    label_box_info[[num_batch]] <- Decode_fun(label_list, anchor_boxs = anchor_boxs, cut_prob = 0.5, cut_overlap = 0.5)
    label_box_info[[num_batch]]$img_ID <- label_box_info[[num_batch]]$img_ID + (num_batch - 1) * dim(img_array)[4]
    
    pred_list <- my_predict(model = model, img = img_array, ctx = mx.gpu())
    pred_box_info[[num_batch]] <- Decode_fun(pred_list, anchor_boxs = anchor_boxs, cut_prob = 0.5, cut_overlap = 0.5)
    pred_box_info[[num_batch]]$img_ID <- pred_box_info[[num_batch]]$img_ID + (num_batch - 1) * dim(img_array)[4]
    
    num_batch <- num_batch + 1
    
  }
  
  label_box_info <- rbindlist(label_box_info) %>% setDF()
  pred_box_info <- rbindlist(pred_box_info) %>% setDF()
  
  label_box_info$IoU <- 0
  pred_box_info$IoU <- 0
  
  for (i in 1:nrow(pred_box_info)) {
    
    sub_label_box_info <- label_box_info[label_box_info$img_ID == pred_box_info[i,'img_ID'], ]
    IoUs <- numeric(nrow(sub_label_box_info))
    
    for (j in 1:nrow(sub_label_box_info)) {
      IoUs[j] <- IoU_function(label = sub_label_box_info[j,2:5], pred = pred_box_info[i,2:5])
    }
    
    pred_box_info$IoU[i] <- max(IoUs)
    label_box_info$IoU[label_box_info$img_ID == pred_box_info[i,'img_ID']][which.max(IoUs)] <- 1
    
  }
  
  obj_names <- unique(pred_box_info$obj_name)
  class_list <- numeric(length(obj_names))
  
  for (i in 1:length(obj_names)) {
    
    obj_label <- pred_box_info[pred_box_info[,1] %in% obj_names[i],'IoU'] > IoU_cut
    
    if (sum(obj_label) == 0) {
      class_list[i] <- 0
    } else {
      num_miss <- sum(label_box_info$IoU == 1 & label_box_info[,1] %in% obj_names[i])
      class_list[i] <- AP_function(vbYreal = c(obj_label, rep(1, num_miss)),
                                   vdYhat = c(pred_box_info[pred_box_info[,1] %in% obj_names[i],'prob'], rep(0, num_miss)))
    }
    
  }
  
  names(class_list) <- obj_names
  
  return(class_list)
  
}

my.yolo_trainer <- function (symbol, Iterator_list, val_iter = NULL,
                             ctx = mx.gpu(), num_round = 5, num_iter = 10,
                             prefix = 'model/yolo model/yolo_v3',
                             Fixed_NAMES = NULL, ARG.PARAMS = NULL, AUX.PARAMS = NULL) {
  
  
  
  lr_decay <- 1
  
  for (k in 1:num_round) {
    
    for (j in 1:length(Iterator_list)) {
      
      message('Start training: round = ', k, ';size = ', j)
      
      #0. Check data shape
      
      Iterator_list[[j]]$reset()
      Iterator_list[[j]]$iter.next()
      my_values <- Iterator_list[[j]]$value()
      input_shape <- lapply(my_values, dim)
      batch_size <- tail(input_shape[[1]], 1)
      
      #1. Build an executor to train model
      
      exec_list <- list(symbol = symbol, fixed.param = Fixed_NAMES, ctx = ctx, grad.req = "write")
      exec_list <- append(exec_list, input_shape)
      my_executor <- do.call(mx.simple.bind, exec_list)
      
      if (k == 1 & j == 1) {
        
        # Set the initial parameters
        
        mx.set.seed(0)
        new_arg <- mxnet:::mx.model.init.params(symbol = symbol,
                                                input.shape = input_shape,
                                                output.shape = NULL,
                                                initializer = mxnet:::mx.init.uniform(0.01),
                                                ctx = ctx)
        
        if (is.null(ARG.PARAMS)) {ARG.PARAMS <- new_arg$arg.params}
        if (is.null(AUX.PARAMS)) {AUX.PARAMS <- new_arg$aux.params}
        
      }
      
      mx.exec.update.arg.arrays(my_executor, ARG.PARAMS, match.name = TRUE)
      mx.exec.update.aux.arrays(my_executor, AUX.PARAMS, match.name = TRUE)
      
      if (!is.null(val_iter)) {map_list <- numeric(num_iter)}
      
      for (i in 1:num_iter) {
        
        my_optimizer <- mx.opt.create(name = "adam", learning.rate = 1e-3/sqrt(lr_decay), beta1 = 0.9, beta2 = 0.999, wd = 1e-4)
        
        my_updater <- mx.opt.get.updater(optimizer = my_optimizer, weights = my_executor$ref.arg.arrays)
        
        Iterator_list[[j]]$reset()
        batch_loss <-  list()
        batch_seq <- 0
        t0 <- Sys.time()
        
        #3. Forward/Backward
        
        while (Iterator_list[[j]]$iter.next()) {
          
          batch_seq <- batch_seq + 1
          
          my_values <- Iterator_list[[j]]$value()
          mx.exec.update.arg.arrays(my_executor, arg.arrays = my_values, match.name = TRUE)
          mx.exec.forward(my_executor, is.train = TRUE)
          mx.exec.backward(my_executor)
          update_args <- my_updater(weight = my_executor$ref.arg.arrays, grad = my_executor$ref.grad.arrays)
          mx.exec.update.arg.arrays(my_executor, update_args, skip.null = TRUE)
          batch_loss[[length(batch_loss) + 1]] <- as.array(my_executor$ref.outputs[[1]])
          
          if (batch_seq %% 50 == 0) {
            message(paste0("Batch [", batch_seq, "] loss =  ", 
                           formatC(mean(unlist(batch_loss)), 4, format = "f"), " (Speed:",
                           formatC(batch_seq * batch_size/as.numeric(Sys.time() - t0, units = 'secs'), format = "f", 2), " samples/sec)"))
          }
          
        }
        
        message(paste0("epoch = ", i,
                       ": loss = ", formatC(mean(unlist(batch_loss)), format = "f", 4),
                       " (Speed: ", formatC(batch_seq * batch_size/as.numeric(Sys.time() - t0, units = 'secs'), format = "f", 2), " samples/sec)"))
        
        my_model <- mxnet:::mx.model.extract.model(symbol = symbol,
                                                   train.execs = list(my_executor))
        
        my_model[[2]] <- append(my_model[[2]], ARG.PARAMS[names(ARG.PARAMS) %in% Fixed_NAMES])
        my_model[[2]] <- my_model[[2]][!names(my_model[[2]]) %in% dim(input_shape)]
        mx.model.save(my_model, prefix, i)
        
        if (!is.null(val_iter) & k != 1) {
          
          ap_list <- model_AP_func(model = my_model, Iterator = val_iter, IoU_cut = 0.5)
          map_list[i] <- mean(ap_list)
          message(paste0("epoch = ", i, ": MAP50 = ", formatC(map_list[i], format = "f", 4)))
          
        }
        
        lr_decay <- lr_decay + 1
        
      }
      
      deleted_list <- list.files(dirname(prefix), pattern = '*.params', full.names = TRUE)
      deleted_list <- deleted_list[!grepl('0000', deleted_list, fixed = TRUE)]
      
      if (!is.null(val_iter) & k != 1) {
        my_model <- mx.model.load(prefix = prefix, iteration = which.max(map_list))
      }
      
      mx.model.save(my_model, prefix, 0)
      file.remove(deleted_list)
      
      ARG.PARAMS <- my_model[[2]]
      AUX.PARAMS <- my_model[[3]]
      
      lr_decay <- sqrt(lr_decay)
      
    }
    
  }
  
  return(my_model)
  
}
