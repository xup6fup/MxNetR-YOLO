
# Libraries

library(xml2)
library(magrittr)
library(jpeg)
library(data.table)
library(OpenImageR)

# Custom function

# Note: this function made some efforts to keep the coordinate system consistent.
# The major challenge is that 'bottomleft' is the original point of "plot" function,
# but the original point of image is 'topleft'

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

# Parameters

resize_size <- 256

# Image_path (Test set)

test_data_dict <- 'data/test/VOCdevkit/VOC2007/'
resize_data_path <- 'data/test_jpg_list.RData'
box_info_path <- 'data/test_info.RData'

# Read annotations

Annotation_list <- list.files(paste0(test_data_dict, 'Annotations'), full.names = TRUE)

# Start to preprocessing 

num_img <- length(Annotation_list)

TEST_BOX_INFOS <- list()
IMG_LIST <- list()

t0 <- Sys.time()

for (i in 1:num_img) {
  
  xml_list <- Annotation_list[i] %>% read_xml() %>% as_list()
  img_path <- paste0(test_data_dict, 'JPEGImages/', xml_list$annotation$filename[[1]])
  object_pos <- which('object' == names(xml_list$annotation))
  
  img <- readImage(img_path)
  
  img_size <- dim(img)[1:2]
  
  if (length(img) == prod(img_size)) {
    img <- array(img, dim = c(img_size, 3))
  }
  
  resized_img <- resizeImage(image = img, width = resize_size, height = resize_size, method = 'bilinear')
  
  resized_img[resized_img < 0] <- 0
  resized_img[resized_img > 1] <- 1
  
  img_box_info <- NULL
  
  for (k in 1:length(object_pos)) {
    
    box_list <- xml_list$annotation[[object_pos[k]]]
    new_img_box_info <- data.frame(obj_name = box_list$name[[1]],
                                   col_left = as.numeric(box_list$bndbox$xmin[[1]])/img_size[2],
                                   col_right = as.numeric(box_list$bndbox$xmax[[1]])/img_size[2],
                                   row_bot = as.numeric(box_list$bndbox$ymax[[1]])/img_size[1],
                                   row_top = as.numeric(box_list$bndbox$ymin[[1]])/img_size[1],
                                   prob = 1,
                                   stringsAsFactors = FALSE)
    
    img_box_info <- rbind(img_box_info, new_img_box_info)
    
  }
  
  TEST_BOX_INFOS[[i]] <- data.frame(img_box_info,
                                    file = xml_list$annotation$filename[[1]],
                                    img_cat = 'test',
                                    img_ID = i,
                                    stringsAsFactors = FALSE)
  
  IMG_LIST[[i]] <- writeJPEG(resized_img)
  
  if (i %% 500 == 0) {
    
    Show_img(resized_img, box_info = img_box_info, show_grid = FALSE)
    
    message(paste0('Current process: ', i, '/', num_img, ' Speed: ',
                   formatC(as.numeric(Sys.time() - t0, units = 'secs')/i, format = 'f', 1), 'sec/img\nEstimated time remaining: ',
                   formatC(as.numeric(Sys.time() - t0, units = 'secs')/i*(num_img - i), format = 'f', 1), 'sec'))
    
  }
  
}

BOX_INFOS <- rbindlist(TEST_BOX_INFOS) %>% setDF()

save(IMG_LIST, file = resize_data_path)
save(BOX_INFOS, file = box_info_path)
