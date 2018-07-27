
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
