
source('pikachu/code/2. Training/1. Encode, Decode & Iterator.R')
source('pikachu/code/2. Training/3. Support functions.R')

# data path (Validation set)

val_img_list_path <- 'pikachu/data/val_img_list.RData'
val_box_info_path <- 'pikachu/data/val_box_info.RData'

# Load data

load(val_img_list_path)
load(val_box_info_path)

# Creat the subset

val_ids <- unique(val_box_info[,'img_id'])
num_val_data <- length(val_ids)
val_batch_size <- 32
num_val_iter <- floor(num_val_data/val_batch_size)
val_input_shape <- c(dim(readJPEG(val_img_list[[1]])), val_batch_size)

# Load well-train model

YOLO_model <- mx.model.load('model/yolo model (pikachu)/yolo_v1', 0)

# Build an executor

val_exec <- mx.simple.bind(symbol = YOLO_model$symbol, data = val_input_shape, ctx = mx.gpu())

mx.exec.update.arg.arrays(val_exec, YOLO_model$arg.params, match.name = TRUE)
mx.exec.update.aux.arrays(val_exec, YOLO_model$aux.params, match.name = TRUE)

# Start to predict

pb <- txtProgressBar(max = num_val_iter, style = 3)

pred_box_info_list <- list()

for (l in 1:num_val_iter) {
  
  idx <- 1:val_batch_size + (l - 1) * val_batch_size
  
  val_img_array <- array(0, dim = val_input_shape)
  
  for (m in 1:length(idx)) {
    
    val_img_array[,,,m] <- readJPEG(val_img_list[[val_ids[idx[m]]]])
    
  }
  
  mx.exec.update.arg.arrays(val_exec, list(data = mx.nd.array(val_img_array)), match.name = TRUE)
  mx.exec.forward(val_exec, is.train = FALSE)
  
  pred_box_info_list[[l]] <- Decode_fun(val_exec$ref.outputs[[1]],
                                        cut_prob = 0.5, cut_overlap = 0.3,
                                        img_id_list = val_ids[idx])
  
  setTxtProgressBar(pb, l)
  
}

close(pb)

# Calculate IoU

pred_box_info <- rbindlist(pred_box_info_list) %>% setDF()

pred_box_info$IoU <- 0
label_box_info <- val_box_info

pb <- txtProgressBar(max = nrow(pred_box_info), style = 3)

for (m in 1:nrow(pred_box_info)) {
  
  sub_label_box_info <- label_box_info[label_box_info[,'img_id'] == pred_box_info[m,'img_id'], ]
  IoUs <- numeric(nrow(sub_label_box_info))
  
  for (n in 1:nrow(sub_label_box_info)) {
    IoUs[n] <- IoU_function(label = sub_label_box_info[n,2:5], pred = pred_box_info[m,2:5])
  }
  
  pred_box_info[m,'IoU'] <- max(IoUs)
  
  setTxtProgressBar(pb, m)
  
}

close(pb)

# Calculate AP

obj_names <- 'pikachu'
ap_list <- numeric(length(obj_names))

for (m in 1:length(obj_names)) {
  
  obj_IoU <- pred_box_info[pred_box_info[,'obj_name'] %in% obj_names[m],'IoU']
  obj_prob <- pred_box_info[pred_box_info[,'obj_name'] %in% obj_names[m],'prob']
  num_obj <- sum(label_box_info[,'obj_name'] == obj_names[m])
  ap_list[m] <- AP_function(obj_IoU = obj_IoU, obj_prob = obj_prob, num_obj = num_obj, IoU_cut = 0.5)
  
}

names(ap_list) <- obj_names
num_obj <- length(obj_names)
if (length(ap_list) < num_obj) {ap_list[(length(ap_list)+1):num_obj] <- 0}

message(paste0("MAP50 = ", formatC(mean(ap_list), format = "f", 4)))
