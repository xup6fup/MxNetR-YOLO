
# Libraries

library(OpenImageR)
library(jpeg)
library(mxnet)

# box_info_path (Training and Validation set)

resize_data_path <- 'data/train_val_jpg_list.RData'
revised_box_info_path <- 'data/train_val_info (yolo v3).RData'
anchor_boxs_path <- 'anchor_boxs (yolo v3).RData'

# Custom function

# Note: this function made some efforts to keep the coordinate system consistent.
# The major challenge is that 'bottomleft' is the original point of "plot" function,
# but the original point of image is 'topleft'
# The Show_img function can help us to encode the bbox info

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

#####################################################################

# Test Encode & Decode function
# You can delete symbol # for running the test

# load(revised_box_info_path)
# load(anchor_boxs_path)
# load(resize_data_path)

# img_id <- 3

# resized_img <- readJPEG(IMG_LIST[[img_id]])

# sub_BOX_INFOS <- BOX_INFOS[BOX_INFOS$img_ID %in% img_id,]

# print(sub_BOX_INFOS)

# Encode_label <- Encode_fun(box_info = sub_BOX_INFOS)
# restore_BOX_INFOS <- Decode_fun(Encode_label, anchor_boxs = anchor_boxs)

# print(Encode_label[[2]][11:13,8:11,26:30,1])

# Show_img(img = resized_img, box_info = sub_BOX_INFOS, n.grid = 16)
# Show_img(img = resized_img, box_info = restore_BOX_INFOS, n.grid = 16)

#####################################################################

# Build an iterator

load(revised_box_info_path)
load(anchor_boxs_path)
load(resize_data_path)

train_ids <- unique(BOX_INFOS[BOX_INFOS$img_cat == 'train','img_ID'])
val_ids <- unique(BOX_INFOS[BOX_INFOS$img_cat == 'val','img_ID'])

my_iterator_core <- function (batch_size, img_size = 256, resize_method = 'nearest',
                              sample = 'train', aug_crop = TRUE, aug_flip = TRUE) {
  
  batch <-  0
  
  if (sample == 'train') {
    
    batch_per_epoch <- floor(length(train_ids)/batch_size)
    
  } else {
    
    batch_per_epoch <- floor(length(val_ids)/batch_size)
    
  }
  
  reset <- function() {batch <<- 0}
  
  iter.next <- function() {
    
    batch <<- batch + 1
    if (batch > batch_per_epoch) {return(FALSE)} else {return(TRUE)}
    
  }
  
  value <- function() {
    
    if (sample == 'train') {id_list <- train_ids} else {id_list <- val_ids}
    
    idx <- 1:batch_size + (batch - 1) * batch_size
    idx[idx > length(id_list)] <- sample(1:(idx[1]-1), sum(idx > length(id_list)))
    idx <- sort(idx)
    
    batch.box_info <- BOX_INFOS[BOX_INFOS$img_ID %in% id_list[idx],]
    
    img_array <- array(0, dim = c(img_size, img_size, 3, batch_size))
    
    for (i in 1:batch_size) {
      
      read_img <- readJPEG(IMG_LIST[[id_list[idx[i]]]])
      
      if (!dim(read_img)[1] == img_size | !dim(read_img)[2] == img_size) {
        
        img_array[,,,i] <- resizeImage(image = read_img, width = img_size, height = img_size, method = resize_method)
        
      } else {
        
        img_array[,,,i] <- read_img
        
      }
      
    }
    
    img_array[img_array < 0] <- 0
    img_array[img_array > 1] <- 1
    
    if (aug_flip) {
      
      original_dim <- dim(img_array)
      
      if (sample(0:1, 1) == 1) {
        
        img_array <- img_array[,original_dim[2]:1,,]
        batch.box_info[,11] <- 1 - batch.box_info[,11]
        dim(img_array) <- original_dim
        
      }
      
    }
    
    if (aug_crop) {
      
      revised_dim <- dim(img_array)
      revised_dim[1:2] <- img_size - 32
      
      random.row <- sample(0:32, 1)
      random.col <- sample(0:32, 1)
      
      img_array <- img_array[random.row+1:(img_size-32),random.col+1:(img_size-32),,]
      dim(img_array) <- revised_dim
      
      batch.box_info[,c(11,12,14)] <- batch.box_info[,c(11,12,14)] * img_size / (img_size - 32)
      batch.box_info[,c(10,13,15)] <- batch.box_info[,c(10,13,15)] * img_size / (img_size - 32)
      
      batch.box_info[,11] <- batch.box_info[,11] - random.col / img_size
      batch.box_info[,10] <- batch.box_info[,10] - random.row / img_size

      batch.box_info[batch.box_info[,11] <= 0,11] <- 0
      batch.box_info[batch.box_info[,11] >= 1,11] <- 1
      batch.box_info[batch.box_info[,10] <= 0,10] <- 0
      batch.box_info[batch.box_info[,10] >= 1,10] <- 1
      
    } 
    
    label <- Encode_fun(box_info = batch.box_info, n.grid = dim(img_array)[1]/c(8, 16, 32))
    for (k in 1:length(label)) {label[[k]] <- mx.nd.array(label[[k]])}
    data <- mx.nd.array(img_array)
    
    return(list(data = data, label1 = label[[1]], label2 = label[[2]], label3 = label[[3]]))
    
  }
  
  return(list(reset = reset, iter.next = iter.next, value = value, batch_size = batch_size, batch = batch))

}

my_iterator_func <- setRefClass("Custom_Iter",
                                fields = c("iter", "batch_size", "img_size", "resize_method", "sample", "aug_crop", "aug_flip"),
                                contains = "Rcpp_MXArrayDataIter",
                                methods = list(
                                  initialize = function(iter, batch_size = 16, img_size = 256, resize_method = 'nearest',
                                                        sample = 'train', aug_crop = TRUE, aug_flip = TRUE){
                                    .self$iter <- my_iterator_core(batch_size = batch_size, img_size = img_size, resize_method = resize_method,
                                                                   sample = sample, aug_crop = aug_crop, aug_flip = aug_flip)
                                    .self
                                  },
                                  value = function(){
                                    .self$iter$value()
                                  },
                                  iter.next = function(){
                                    .self$iter$iter.next()
                                  },
                                  reset = function(){
                                    .self$iter$reset()
                                  },
                                  finalize=function(){
                                  }
                                )
)

#####################################################################

# Test iterator function
# You can delete symbol # for running the test

# my_iter <- my_iterator_func(iter = NULL, batch_size = 16, img_size = 320, resize_method = 'nearest',
#                             sample = 'train', aug_crop = TRUE, aug_flip = TRUE)

# my_iter$reset()

# t0 <- Sys.time()
 
# for (i in 1:sample(20, 1)) {my_iter$iter.next()}
 
# test <- my_iter$value()
 
# print(Sys.time() - t0)
 
# img_seq <- sample(16, 1)

# iter_img <- as.array(test$data)[,,,img_seq]
 
# If you use 'aug_crop = TRUE', you need to revise the anchor_boxs
 
# revised_anchor_boxs <- anchor_boxs
# revised_anchor_boxs[,1:2] <- revised_anchor_boxs[,1:2] * 320 / (320 - 32)
 
# label_list <- list(test$label1, test$label2, test$label3)
# iter_box_info <- Decode_fun(label_list, anchor_boxs = revised_anchor_boxs)

# Show_img(img = iter_img, box_info = iter_box_info[iter_box_info$img_ID == img_seq,], show_grid = FALSE, n.grid = 7)


