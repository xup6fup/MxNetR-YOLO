
# Libraries

library(OpenImageR)
library(jpeg)
library(mxnet)
library(imager)

# data path (Validation set)

val_img_list_path <- 'pikachu/data/val_img_list.RData'
val_box_info_path <- 'pikachu/data/val_box_info.RData'

if (!'val_img_list' %in% ls()) {
  
  # Load data
  
  load(val_img_list_path)
  load(val_box_info_path)
  
}


# Custom function

# Note: this function made some efforts to keep the coordinate system consistent.
# The major challenge is that 'bottomleft' is the original point of "plot" function,
# but the original point of image is 'topleft'
# The Show_img function can help us to encode the bbox info

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

Show_img <- function (img, box_info = NULL, show_prob = FALSE, col_bbox = '#FFFFFF00', col_label = '#FF0000FF',
                      show_grid = FALSE, n.grid = 8, col_grid = '#0000FFFF') {
  
  require(imager)
  
  par(mar = rep(0, 4))
  plot(NA, xlim = c(0.03, 0.97), ylim = c(0.97, 0.03), xaxt = "n", yaxt = "n", bty = "n")
  img <- (img - min(img))/(max(img) - min(img))
  img <- as.raster(img)
  rasterImage(img, 0, 1, 1, 0, interpolate=FALSE)
  
  if (!is.null(box_info)) {
    box_info[box_info[,2] < 0, 2] <- 0
    box_info[box_info[,3] > 1, 3] <- 1
    box_info[box_info[,4] > 1, 4] <- 1
    box_info[box_info[,5] < 0, 5] <- 0
  }
  
  box_info <- box_info[order(box_info[,6]),]
  
  if (!is.null(box_info)) {
    for (i in 1:nrow(box_info)) {
      if (is.null(box_info$col[i])) {COL_LABEL <- col_label} else {COL_LABEL <- box_info$col[i]}
      if (show_prob) {
        TEXT <- paste0(box_info[i,1], ' (', formatC(box_info[i,6]*100, 0, format = 'f'), '%)')
      } else {
        TEXT <- box_info[i,1]
      }
      size <- max(box_info[i,3] - box_info[i,2], 0.05)
      rect(xleft = box_info[i,2], xright = box_info[i,2] + 0.04*sqrt(size)*nchar(TEXT),
           ybottom = box_info[i,5] + 0.08*sqrt(size), ytop = box_info[i,5],
           col = COL_LABEL, border = COL_LABEL, lwd = 0)
      text(x = box_info[i,2] + 0.02*sqrt(size) * nchar(TEXT),
           y = box_info[i,5] + 0.04*sqrt(size),
           labels = TEXT,
           col = 'white', cex = 1.5*sqrt(size), font = 2)
      rect(xleft = box_info[i,2], xright = box_info[i,3],
           ybottom = box_info[i,4], ytop = box_info[i,5],
           col = col_bbox, border = COL_LABEL, lwd = 5*sqrt(size))
    }
  }
  
  if (show_grid) {
    for (i in 1:n.grid) {
      if (i != n.grid) {
        abline(a = i/n.grid, b = 0, col = col_grid, lwd = 12/n.grid)
        abline(v = i/n.grid, col = col_grid, lwd = 12/n.grid)
      }
      for (j in 1:n.grid) {
        text((i-0.5)/n.grid, (j-0.5)/n.grid, paste0('(', j, ', ', i, ')'), col = col_grid, cex = 8/n.grid)
      }
    }
  }
  
}

Decode_fun <- function (encode_array, cut_prob = 0.5, cut_overlap = 0.3,
                        obj_name = 'pikachu',
                        obj_col = '#FF0000FF',
                        img_id_list = NULL) {
  
  num_img <- dim(encode_array)[4]
  num_feature <- length(obj_name) + 5
  pos_start <- (0:(dim(encode_array)[3]/num_feature-1)*num_feature)
  
  box_info <- NULL
  
  # Decoding
  
  for (j in 1:num_img) {
    
    sub_box_info <- NULL
    
    for (i in 1:length(pos_start)) {
      
      sub_encode_array <- as.array(encode_array)[,,pos_start[i]+1:num_feature,j]
      
      pos_over_cut <- which(sub_encode_array[,,1] >= cut_prob)
      
      if (length(pos_over_cut) >= 1) {
        
        pos_over_cut_row <- pos_over_cut %% dim(sub_encode_array)[1]
        pos_over_cut_row[pos_over_cut_row == 0] <- dim(sub_encode_array)[1]
        pos_over_cut_col <- ceiling(pos_over_cut/dim(sub_encode_array)[1])
        
        for (l in 1:length(pos_over_cut)) {
          
          encode_vec <- sub_encode_array[pos_over_cut_row[l],pos_over_cut_col[l],]
          
          if (encode_vec[2] < 0) {encode_vec[2] <- 0}
          if (encode_vec[2] > 1) {encode_vec[2] <- 1}
          if (encode_vec[3] < 0) {encode_vec[3] <- 0}
          if (encode_vec[3] > 1) {encode_vec[3] <- 1}
          
          center_row <- (encode_vec[2] + (pos_over_cut_row[l] - 1))/dim(sub_encode_array)[1]
          center_col <- (encode_vec[3] + (pos_over_cut_col[l] - 1))/dim(sub_encode_array)[2]
          width <- exp(encode_vec[4])/dim(sub_encode_array)[2]
          height <- exp(encode_vec[5])/dim(sub_encode_array)[1]
          
          if (is.null(img_id_list)) {new_img_id <- j} else {new_img_id <- img_id_list[j]}
          
          new_box_info <- data.frame(obj_name = obj_name[which.max(encode_vec[-c(1:5)])],
                                     col_left = center_col-width/2,
                                     col_right = center_col+width/2,
                                     row_bot = center_row+height/2,
                                     row_top = center_row-height/2,
                                     prob = encode_vec[1],
                                     img_id = new_img_id,
                                     col = obj_col[which.max(encode_vec[-c(1:5)])],
                                     stringsAsFactors = FALSE)
          
          sub_box_info <- rbind(sub_box_info, new_box_info)
          
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
                
                overlap_prob <- IoU_function(label = obj_sub_box_info[m,2:5], pred = obj_sub_box_info[n,2:5])
                
                overlap_width <- min(obj_sub_box_info[m,3], obj_sub_box_info[n,3]) - max(obj_sub_box_info[m,2], obj_sub_box_info[n,2])
                overlap_height <- min(obj_sub_box_info[m,4], obj_sub_box_info[n,4]) - max(obj_sub_box_info[m,5], obj_sub_box_info[n,5])
                
                if (overlap_prob >= cut_overlap) {
                  
                  overlap_seq <- c(overlap_seq, m)
                  
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

# Load well-train model

YOLO_model <- mx.model.load('model/yolo model (pikachu)/yolo_v1', 0)

# Read jpg and resize

used_img_id <- 5

img <- readJPEG(val_img_list[[used_img_id]])
img <- resizeImage(image = img, width = 256, height = 256, method = 'bilinear')
dim(img) <- c(256, 256, 3, 1)

# Predict and decode

pred_out <- mxnet:::predict.MXFeedForwardModel(model = YOLO_model, X = img)
pred_box_info <- Decode_fun(pred_out, cut_prob = 0.5, cut_overlap = 0.3)

# Show image

Show_img(img = img[,,,1], box_info = pred_box_info, show_prob = TRUE, show_grid = FALSE)
