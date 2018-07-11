
# Libraries

library(OpenImageR)
library(mxnet)

# box_info_path (Training and Validation set)

revised_box_info_path <- 'data/train_val_info (for iterater).RData'
resized_img_path <- 'data/resize_img/'

# Custom function

# Note: this function made some efforts to keep the coordinate system consistent.
# The major challenge is that 'bottomleft' is the original point of "plot" function,
# but the original point of image is 'topleft'
# The Show_img function can help us to encode the bbox info

Show_img <- function (img, box_info = NULL, col_bbox = '#00A80050', col_label = '#00A0A0FF',
                      show_grid = TRUE, n.grid = 8, col_grid = '#FF0000FF') {
  
  require(imager)
  
  par(mar = rep(0, 4))
  plot(NA, xlim = c(0.04, 0.96), ylim = c(0.96, 0.04), xaxt = "n", yaxt = "n", bty = "n")
  img = as.raster(img)
  rasterImage(img, 0, 1, 1, 0, interpolate=FALSE)
  
  if (!is.null(box_info)) {
    for (i in 1:nrow(box_info)) {
      text(x = (box_info[i,2] + box_info[i,3])/2, y = box_info[i,5],
           labels = paste0(box_info[i,1], ' (', formatC(box_info[i,6]*100, 0, format = 'f'), '%)'),
           offset = 0.3, pos = 1, col = col_label, font = 2)
      rect(xleft = box_info[i,2], xright = box_info[i,3],
           ybottom = box_info[i,4], ytop = box_info[i,5],
           col = col_bbox, border = col_label, lwd = 1.5)
    }
  }
  
  if (show_grid) {
    for (i in 1:n.grid) {
      if (i != n.grid) {
        abline(a = i/n.grid, b = 0, col = col_grid, lwd = 1.5)
        abline(v = i/n.grid, col = col_grid, lwd = 1.5)
      }
      for (j in 1:n.grid) {
        text((i-0.5)/n.grid, (j-0.5)/n.grid, paste0('(', j, ', ', i, ')'), col = 'red')
      }
    }
  }
  
}

Encode_fun <- function (box_info, n.grid = c(32, 16, 8),
                        obj_name = c('person', 'bird', 'cat', 'cow', 'dog',
                                     'horse', 'sheep', 'aeroplane', 'bicycle',
                                     'boat', 'bus', 'car', 'motorbike', 'train',
                                     'bottle', 'chair', 'diningtable', 'pottedplant',
                                     'sofa', 'tvmonitor')) {
  
  img_IDs <- unique(box_info$img_ID)
  num_pred <- 5 + length(obj_name)
  
  out_array_list <- list()
  
  for (k in 1:length(n.grid)) {
    
    out_array_list[[k]] <- array(0, dim = c(n.grid[k], n.grid[k], 3 * num_pred, length(img_IDs)))
    
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
          
          out_array_list[[k]][center_row,center_col,(rescale_box_info$seq[i]-1)*num_pred+1,j] <- 1
          out_array_list[[k]][center_row,center_col,(rescale_box_info$seq[i]-1)*num_pred+2,j] <- rescale_box_info$bbox_center_row[i] %% 1
          out_array_list[[k]][center_row,center_col,(rescale_box_info$seq[i]-1)*num_pred+3,j] <- rescale_box_info$bbox_center_col[i] %% 1
          out_array_list[[k]][center_row,center_col,(rescale_box_info$seq[i]-1)*num_pred+4,j] <- log(rescale_box_info$bbox_width[i]/rescale_box_info$anchor_width[i])
          out_array_list[[k]][center_row,center_col,(rescale_box_info$seq[i]-1)*num_pred+5,j] <- log(rescale_box_info$bbox_height[i]/rescale_box_info$anchor_height[i])
          out_array_list[[k]][center_row,center_col,(rescale_box_info$seq[i]-1)*num_pred+5+which(obj_name %in% rescale_box_info$obj_name[i]),j] <- 1 
          
        }
        
      }
      
    }
    
  }
  
  return(out_array_list)
  
}

# Test Encode function

load(revised_box_info_path)

img_id <- 2

load(paste0(resized_img_path, img_id, '.RData'))
Show_img(img = resized_img, box_info = BOX_INFOS[BOX_INFOS$img_ID == img_id,1:6])

print(BOX_INFOS[BOX_INFOS$img_ID == img_id,])
Encode_label <- Encode_fun(box_info = BOX_INFOS[BOX_INFOS$img_ID == img_id,])

print(Encode_label[[3]][,,1:5,1])

# Build an iterator

my_iterator_core <- function (batch_size, aug = TRUE) {
  
  batch = 0
  batch_per_epoch = nrow(Train.box_info)/batch_size
  
  reset = function() {batch <<- 0}
  
  iter.next = function() {
    batch <<- batch+1
    if (batch > batch_per_epoch) {return(FALSE)} else {return(TRUE)}
  }
  
  value = function() {
    
    idx = 1:batch_size + (batch - 1) * batch_size
    idx[idx > nrow(Train.box_info)] = sample(1:nrow(Train.box_info), sum(idx > nrow(Train.box_info)))
    
    batch.box_info <- Train.box_info[idx,]
    
    if (aug) {
      
      random.row <- sample(0:32, 1)
      random.col <- sample(0:32, 1)
      
      data = mx.nd.array(Train.img_array[random.row+1:224,random.col+1:224,,idx])
      batch.box_info[,3:4] <- (1 - (1 - batch.box_info[,3:4])*8/7 + random.row/256)
      batch.box_info[,1:2] <- batch.box_info[,1:2]*8/7 - random.col/256
      
    } else {
      
      data = mx.nd.array(Train.img_array[,,,idx])
      
    }
    
    label = mx.nd.array(Encode_fun(box_info = batch.box_info))
    
    return(list(data = data, label = label))
  }
  
  return(list(reset = reset, iter.next = iter.next, value = value, batch_size = batch_size, batch = batch))
}