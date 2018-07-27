
# Libraries

library(jsonlite)
library(jpeg)
library(magrittr)
library(OpenImageR)

# Custom function

# Note: this function made some efforts to keep the coordinate system consistent.
# The major challenge is that 'bottomleft' is the original point of "plot" function,
# but the original point of image is 'topleft'

Show_img <- function (img, box_info = NULL, show_prob = FALSE, col_bbox = '#FFFFFF00', col_label = '#FF0000FF',
                      show_grid = FALSE, n.grid = 8, col_grid = '#0000FFFF') {
  
  require(imager)
  
  par(mar = rep(0, 4))
  plot(NA, xlim = c(0.03, 0.97), ylim = c(0.97, 0.03), xaxt = "n", yaxt = "n", bty = "n")
  img <- (img - min(img))/(max(img) - min(img))
  img <- as.raster(img)
  rasterImage(img, 0, 1, 1, 0, interpolate=FALSE)
  
  box_info[box_info[,2] < 0, 2] <- 0
  box_info[box_info[,3] > 1, 3] <- 1
  box_info[box_info[,4] > 1, 4] <- 1
  box_info[box_info[,5] < 0, 5] <- 0
  
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
        text((i-0.5)/n.grid, (j-0.5)/n.grid, paste0('(', j, ', ', i, ')'), col = 'red', cex = 8/n.grid)
      }
    }
  }
  
}

# Image_path (Validation set)

val_data_dict <- 'pikachu/data/val/'
val_annot_path <- 'pikachu/data/val_label.txt'

img_list_path <- 'pikachu/data/val_img_list.RData'
box_info_path <- 'pikachu/data/val_box_info.RData'

# Read annotations

val_box_info <- read.table(val_annot_path, header = TRUE, stringsAsFactors = FALSE)
save(val_box_info, file = box_info_path)

# Start to preprocessing (image)

val_img_list <- list()

val_img_id <- unique(val_box_info$img_id)
num_img <- length(val_img_id)

t0 <- Sys.time()

for (i in 1:num_img) {
  
  img <- try(readJPEG(paste0(val_data_dict, val_img_id[i], '.jpeg')), silent = TRUE)
  
  if (is.array(img)) {
    
    val_img_list[[i]] <- writeJPEG(img, quality = 1)
    
    if (i %% 20 == 0) {
      Show_img(img = readJPEG(val_img_list[[i]]),
               box_info = val_box_info[val_box_info[,'img_id'] == val_img_id[i],])
      
      message(paste0('Current process: ', i, '/', num_img, ' Speed: ',
                     formatC(as.numeric(Sys.time() - t0, units = 'secs')/i, format = 'f', 1), 'sec/img\nEstimated time remaining: ',
                     formatC(as.numeric(Sys.time() - t0, units = 'secs')/i*(num_img - i), format = 'f', 1), 'sec'))
    }
    
  }
  
}

save(val_img_list, file = img_list_path)
