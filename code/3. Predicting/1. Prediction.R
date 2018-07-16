
# Custom predict function

Show_img <- function (img, box_info = NULL, col_bbox = '#FFFFFF00', col_label = '#00A0A0FF',
                      show_grid = TRUE, n.grid = 8, col_grid = '#FF0000FF') {
  
  require(imager)
  
  par(mar = rep(0, 4))
  plot(NA, xlim = c(0.04, 0.96), ylim = c(0.96, 0.04), xaxt = "n", yaxt = "n", bty = "n")
  img <- (img - min(img))/(max(img) - min(img))
  img <- as.raster(img)
  rasterImage(img, 0, 1, 1, 0, interpolate=FALSE)
  
  if (!is.null(box_info)) {
    for (i in 1:nrow(box_info)) {
      if (is.null(box_info$col[i])) {COL_LABEL <- col_label} else {COL_LABEL <- box_info$col[i]}
      text(x = (box_info[i,2] + box_info[i,3])/2, y = box_info[i,5],
           labels = paste0(box_info[i,1], ' (', formatC(box_info[i,6]*100, 0, format = 'f'), '%)'),
           offset = 0.3, pos = 1, col = COL_LABEL, font = 2)
      rect(xleft = box_info[i,2], xright = box_info[i,3],
           ybottom = box_info[i,4], ytop = box_info[i,5],
           col = col_bbox, border = COL_LABEL, lwd = 1.5)
    }
  }
  
  if (show_grid) {
    for (i in 1:n.grid) {
      if (i != n.grid) {
        abline(a = i/n.grid, b = 0, col = col_grid, lwd = 12/n.grid)
        abline(v = i/n.grid, col = col_grid, lwd = 12/n.grid)
      }
      for (j in 1:n.grid) {
        text((i-0.5)/n.grid, (j-0.5)/n.grid, paste0('(', j, ', ', i, ')'), col = 'red', cex = 8/n.grid)
      }
    }
  }
  
}

Encode_fun <- function (box_info, n.grid = c(32, 16, 8), eps = 1e-8, n.anchor = 3,
                        obj_name = c('person', 'bird', 'cat', 'cow', 'dog',
                                     'horse', 'sheep', 'aeroplane', 'bicycle',
                                     'boat', 'bus', 'car', 'motorbike', 'train',
                                     'bottle', 'chair', 'diningtable', 'pottedplant',
                                     'sofa', 'tvmonitor')) {
  
  img_IDs <- unique(box_info$img_ID)
  num_pred <- 5 + length(obj_name)
  
  out_array_list <- list()
  
  for (k in 1:length(n.grid)) {
    
    out_array_list[[k]] <- array(0, dim = c(n.grid[k], n.grid[k], n.anchor * num_pred, length(img_IDs)))
    
  }
  
  for (j in 1:length(img_IDs)) {
    
    sub_box_info <- box_info[box_info$img_ID == img_IDs[j],]
    
    for (k in 1:length(n.grid)) {
      
      if (k %in% sub_box_info$lvl) {
        
        rescale_box_info <- sub_box_info[sub_box_info$lvl == k,c(1, 10:15, 17:18)]
        rescale_box_info[,2:7] <- rescale_box_info[,2:7] * n.grid[k]
        
        for (i in 1:nrow(rescale_box_info)) {
          
          center_row <- ceiling(rescale_box_info$bbox_center_row[i])
          center_col <- ceiling(rescale_box_info$bbox_center_col[i])
          
          row_related_pos <- rescale_box_info$bbox_center_row[i] %% 1
          row_related_pos[row_related_pos == 0] <- 1
          col_related_pos <- rescale_box_info$bbox_center_col[i] %% 1
          col_related_pos[col_related_pos == 0] <- 1
          
          out_array_list[[k]][center_row,center_col,(rescale_box_info$seq[i]-1)*num_pred+1,j] <- 1
          out_array_list[[k]][center_row,center_col,(rescale_box_info$seq[i]-1)*num_pred+2,j] <- row_related_pos
          out_array_list[[k]][center_row,center_col,(rescale_box_info$seq[i]-1)*num_pred+3,j] <- col_related_pos
          out_array_list[[k]][center_row,center_col,(rescale_box_info$seq[i]-1)*num_pred+4,j] <- log(rescale_box_info$bbox_width[i]/rescale_box_info$anchor_width[i] + eps)
          out_array_list[[k]][center_row,center_col,(rescale_box_info$seq[i]-1)*num_pred+5,j] <- log(rescale_box_info$bbox_height[i]/rescale_box_info$anchor_height[i] + eps)
          out_array_list[[k]][center_row,center_col,(rescale_box_info$seq[i]-1)*num_pred+5+which(obj_name %in% rescale_box_info$obj_name[i]),j] <- 1 
          
        }
        
      }
      
    }
    
  }
  
  return(out_array_list)
  
}

Decode_fun <- function (encode_array_list, anchor_boxs,
                        cut_prob = 0.5, cut_overlap = 0.5,
                        obj_name = c('person', 'bird', 'cat', 'cow', 'dog',
                                     'horse', 'sheep', 'aeroplane', 'bicycle',
                                     'boat', 'bus', 'car', 'motorbike', 'train',
                                     'bottle', 'chair', 'diningtable', 'pottedplant',
                                     'sofa', 'tvmonitor'),
                        obj_col = c('#FF0000FF', '#00FF00FF', '#00FF00FF', '#00FF00FF', '#00FF00FF',
                                    '#00FF00FF', '#00FF00FF', '#0000FFFF', '#0000FFFF', '#0000FFFF',
                                    '#0000FFFF', '#0000FFFF', '#0000FFFF', '#0000FFFF', '#FFFF00FF',
                                    '#FFFF00FF', '#FFFF00FF', '#FFFF00FF', '#FFFF00FF', '#FFFF00FF')) {
  
  num_list <- length(encode_array_list)
  num_img <- dim(encode_array_list[[1]])[4]
  num_feature <- length(obj_name) + 5
  pos_start <- (0:(dim(encode_array_list[[1]])[3]/num_feature-1)*num_feature)
  
  box_info <- NULL
  
  # Decoding
  
  for (j in 1:num_img) {
    
    sub_box_info <- NULL
    
    for (k in 1:num_list) {
      
      for (i in 1:length(pos_start)) {
        
        sub_encode_array <- as.array(encode_array_list[[k]])[,,pos_start[i]+1:num_feature,j]
        
        pos_over_cut <- which(sub_encode_array[,,1] >= cut_prob)
        
        if (length(pos_over_cut) >= 1) {
          
          pos_over_cut_row <- pos_over_cut %% dim(sub_encode_array)[1]
          pos_over_cut_row[pos_over_cut_row == 0] <- dim(sub_encode_array)[1]
          pos_over_cut_col <- ceiling(pos_over_cut/dim(sub_encode_array)[1])
          anchor_box <- anchor_boxs[anchor_boxs$lvl == k & anchor_boxs$seq == i, 1:2]
          
          for (l in 1:length(pos_over_cut)) {
            
            encode_vec <- sub_encode_array[pos_over_cut_row[l],pos_over_cut_col[l],]
            
            if (encode_vec[2] < 0) {encode_vec[2] <- 0}
            if (encode_vec[2] > 1) {encode_vec[2] <- 1}
            if (encode_vec[3] < 0) {encode_vec[3] <- 0}
            if (encode_vec[3] > 1) {encode_vec[3] <- 1}
            
            center_row <- (encode_vec[2] + (pos_over_cut_row[l] - 1))/dim(sub_encode_array)[1]
            center_col <- (encode_vec[3] + (pos_over_cut_col[l] - 1))/dim(sub_encode_array)[2]
            width <- exp(encode_vec[4]) * anchor_box[1,1]
            height <- exp(encode_vec[5]) * anchor_box[1,2]
            
            new_box_info <- data.frame(obj_name = obj_name[which.max(encode_vec[-c(1:5)])],
                                       col_left = center_col-width/2,
                                       col_right = center_col+width/2,
                                       row_bot = center_row+height/2,
                                       row_top = center_row-height/2,
                                       prob = encode_vec[1],
                                       img_ID = j,
                                       col = obj_col[which.max(encode_vec[-c(1:5)])],
                                       stringsAsFactors = FALSE)
            
            sub_box_info <- rbind(sub_box_info, new_box_info)
            
          }
          
        }
        
      }
      
    }
    
    if (!is.null(sub_box_info)) {
      
      # Remove overlapping
      
      sub_box_info <- sub_box_info[order(sub_box_info$prob, decreasing = TRUE),]
      
      for (obj in unique(sub_box_info$obj_name)) {
        
        obj_sub_box_info <- sub_box_info[sub_box_info$obj_name == obj,]
        
        if (nrow(obj_sub_box_info) == 1) {
          
          box_info <- rbind(box_info, obj_sub_box_info)
          
        } else {
          
          overlap_seq <- NULL
          
          for (m in 2:nrow(obj_sub_box_info)) {
            
            for (n in 1:(m-1)) {
              
              if (!n %in% overlap_seq) {
                
                overlap_width <- min(obj_sub_box_info[m,3], obj_sub_box_info[n,3]) - max(obj_sub_box_info[m,2], obj_sub_box_info[n,2])
                overlap_height <- min(obj_sub_box_info[m,4], obj_sub_box_info[n,4]) - max(obj_sub_box_info[m,5], obj_sub_box_info[n,5])
                
                if (overlap_width > 0 & overlap_height > 0) {
                  
                  old_size <- (obj_sub_box_info[n,3]-obj_sub_box_info[n,2])*(obj_sub_box_info[n,4]-obj_sub_box_info[n,5])
                  new_size <- (obj_sub_box_info[m,3]-obj_sub_box_info[m,2])*(obj_sub_box_info[m,4]-obj_sub_box_info[m,5])
                  overlap_size <- overlap_width * overlap_height
                  
                  if (overlap_size/min(old_size, new_size) >= cut_overlap) {
                    
                    overlap_seq <- c(overlap_seq, m)
                    
                  }
                  
                }
                
              }
              
            }
            
          }
          
          if (!is.null(overlap_seq)) {
            
            obj_sub_box_info <- obj_sub_box_info[-overlap_seq,]
            
          }
          
          box_info <- rbind(box_info, obj_sub_box_info)
          
        }
        
      }
      
    }
    
  }
  
  return(box_info)
  
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

# Load well-train model

YOLO_model <- mx.model.load('model/yolo model/yolo_v3', 0)
load('anchor_boxs (yolo v3).RData')

# You can select to use train set or testing set (if you have conducted all codes for training)

# resize_test_data_path <- 'data/test_jpg_list.RData'
# resize_train_data_path <- 'data/train_val_jpg_list.RData'

# load(resize_test_data_path)
# load(resize_train_data_path)

# Read jpg and resize

img <- readJPEG('test_img.jpeg')
img <- resizeImage(image = img, width = 256, height = 256, method = 'bilinear')
dim(img) <- c(256, 256, 3, 1)

# Predict and decode

pred_list <- my_predict(model = YOLO_model, img = img, ctx = mx.gpu())
pred_box_info <- Decode_fun(pred_list, anchor_boxs = anchor_boxs, cut_prob = 0.5, cut_overlap = 0.5)

# Show image

Show_img(img = img[,,,1], box_info = pred_box_info, show_grid = FALSE)
