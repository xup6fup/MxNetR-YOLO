
# box_info_path (Training and Validation set)

box_info_path <- 'data/train_val_info.RData'

# Start to define anchor boxes 

load(box_info_path)

anchor_box_info <- data.frame(width = (BOX_INFOS[,3] - BOX_INFOS[,2]),
                              height = (BOX_INFOS[,4] - BOX_INFOS[,5]),
                              stringsAsFactors = FALSE)

kmean_model <- kmeans(x = anchor_box_info, centers = 9, iter.max = 10)

anchor_boxs <- as.data.frame(kmean_model$centers, stringsAsFactors = FALSE)
anchor_boxs$rank <- rank(anchor_boxs[,1] * anchor_boxs[,2])
anchor_boxs$lvl <- ceiling(anchor_boxs$rank / 3)
anchor_boxs$seq <- anchor_boxs$rank %% 3 + 1
anchor_boxs$col <- rainbow(9)[anchor_boxs$rank]

plot(anchor_box_info$width, anchor_box_info$height, pch = 19, cex = 0.5,
     col = anchor_boxs$col[kmean_model$cluster])
