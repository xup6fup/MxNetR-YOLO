
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

# Image_path (Training set)

train_data_dict <- 'pikachu/data/train/'
train_annot_path <- 'pikachu/data/train_label.txt'

img_list_path <- 'coco2017/data/train_img_list.RData'
box_info_path <- 'coco2017/data/train_box_info.RData'

# Read annotations

annotation_list <- read.table(train_annot_path, header = TRUE, stringsAsFactors = FALSE)

Show_img(readJPEG(paste0(train_data_dict, '928.jpeg')), box_info = annotation_list[928,])

