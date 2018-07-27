
# Libraries

library(mxnet)
library(data.table)
library(magrittr)

# Custom functions

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

# Based on https://github.com/rafaelpadilla/Object-Detection-Metrics

AP_function <- function (obj_IoU, obj_prob, num_obj, IoU_cut = 0.5) {
  
  sort_obj_IoU <- obj_IoU[order(obj_prob, decreasing=TRUE)]
  pred_postive <- sort_obj_IoU > IoU_cut
  
  cum_TP <- cumsum(pred_postive)
  
  P_list <- cum_TP * pred_postive / seq_along(pred_postive)
  P_list <- P_list[P_list!=0]
  
  while (sum(diff(P_list) > 0) >= 1) {
    diff_P_list <- diff(P_list)
    diff_P_list[diff_P_list < 0] <- 0
    P_list <- P_list + c(diff_P_list, 0)
  }
  
  return(sum(P_list)/num_obj)
  
}

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
                               prefix = 'model/yolo model (pikachu)/yolo_v1',
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
