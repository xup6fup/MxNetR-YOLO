
source('voc2007/code/2. Training/1. Encode, Decode & Iterator.R')
source('voc2007/code/2. Training/3. Support functions.R')

resize_test_data_path <- 'voc2007/data/test_jpg_list.RData'
revised_test_box_info_path <- 'voc2007/data/test_info.RData'
IoU_cut <- 0.5

YOLO_model <- mx.model.load('model/yolo model/yolo_v3', 0)

load(resize_test_data_path)
load(revised_test_box_info_path)

num_img <- length(IMG_LIST)
seq_img <- 1
pred_box_info <- list()

pb <- txtProgressBar(max = ceiling(num_img/100), style = 3)

for (i in 1:ceiling(num_img/100)) {
  
  img_array <- array(0, dim = c(256, 256, 3, 100))
  
  for (j in 1:100) {
    if (seq_img > num_img) {
      img_array <- img_array[,,,1:(j-1)]
      break
    } else {
      img_array[,,,j] <- readJPEG(IMG_LIST[[seq_img]])
      seq_img <- seq_img + 1
    }
  }
  
  pred_list <- my_predict(model = YOLO_model, img = img_array, ctx = mx.gpu())
  pred_box_info[[i]] <- Decode_fun(pred_list, anchor_boxs = anchor_boxs, cut_prob = 0.5, cut_overlap = 0.5)
  pred_box_info[[i]]$img_ID <- pred_box_info[[i]]$img_ID + (i - 1) * 100
 
  setTxtProgressBar(pb, i)
   
}

close(pb)

pred_box_info <- rbindlist(pred_box_info) %>% setDF()

BOX_INFOS$IoU <- 0
pred_box_info$IoU <- 0

for (i in 1:nrow(pred_box_info)) {
  
  sub_label_box_info <- BOX_INFOS[BOX_INFOS$img_ID == pred_box_info[i,'img_ID'], ]
  IoUs <- numeric(nrow(sub_label_box_info))
  
  for (j in 1:nrow(sub_label_box_info)) {
    IoUs[j] <- IoU_function(label = sub_label_box_info[j,2:5], pred = pred_box_info[i,2:5])
  }
  
  pred_box_info$IoU[i] <- max(IoUs)
  BOX_INFOS$IoU[BOX_INFOS$img_ID == pred_box_info[i,'img_ID']][which.max(IoUs)] <- 1
  
}

obj_names <- unique(pred_box_info$obj_name)
class_list <- numeric(length(obj_names))

for (i in 1:length(obj_names)) {
  
  obj_label <- pred_box_info[pred_box_info[,1] %in% obj_names[i],'IoU'] > IoU_cut
  
  if (sum(obj_label) == 0) {
    class_list[i] <- 0
  } else {
    num_miss <- sum(BOX_INFOS$IoU == 1 & BOX_INFOS[,1] %in% obj_names[i])
    class_list[i] <- AP_function(vbYreal = c(obj_label, rep(1, num_miss)),
                                 vdYhat = c(pred_box_info[pred_box_info[,1] %in% obj_names[i],'prob'], rep(0, num_miss)))
  }
  
}

names(class_list) <- obj_names

message(paste0("test-MAP50 = ", formatC(mean(class_list), format = "f", 4)))
