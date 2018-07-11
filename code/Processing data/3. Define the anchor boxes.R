
# box_info_path (Training and Validation set)

original_box_info_path <- 'data/train_val_info.RData'
revised_box_info_path <- 'data/train_val_info (for iterater).RData'
anchor_boxs_path <- 'anchor_boxs.RData'

# Start to define anchor boxes 

load(original_box_info_path)

anchor_box_info <- data.frame(width = log(BOX_INFOS[,3] - BOX_INFOS[,2]),
                              height = log(BOX_INFOS[,4] - BOX_INFOS[,5]),
                              stringsAsFactors = FALSE)

kmean_model <- kmeans(x = anchor_box_info, centers = 9, iter.max = 10)

anchor_boxs <- as.data.frame(kmean_model$centers, stringsAsFactors = FALSE)
anchor_boxs$width <- exp(anchor_boxs$width)
anchor_boxs$height <- exp(anchor_boxs$height)
anchor_boxs$rank <- rank(anchor_boxs[,1] * anchor_boxs[,2])
anchor_boxs$lvl <- ceiling(anchor_boxs$rank / 3)
anchor_boxs$seq <- anchor_boxs$rank %% 3 + 1
anchor_boxs$col <- rainbow(9)[anchor_boxs$rank]

# Visualization

par(mar = c(5, 4, 4, 2))

plot(exp(anchor_box_info$width), exp(anchor_box_info$height), pch = 19, cex = 0.5,
     col = anchor_boxs$col[kmean_model$cluster], 
     xlab = 'Width', ylab = 'Height', main = 'Anchor box clusters')

# Add anchor box info to BOX_INFOS

BOX_INFOS$bbox_center_row <- (BOX_INFOS[,4] + BOX_INFOS[,5])/2
BOX_INFOS$bbox_center_col <- (BOX_INFOS[,2] + BOX_INFOS[,3])/2
BOX_INFOS$bbox_width <- exp(anchor_box_info$width)
BOX_INFOS$bbox_height <- exp(anchor_box_info$height)
BOX_INFOS$anchor_width <- anchor_boxs$width[kmean_model$cluster]
BOX_INFOS$anchor_height <- anchor_boxs$height[kmean_model$cluster]
BOX_INFOS$rank <- anchor_boxs$rank[kmean_model$cluster]
BOX_INFOS$lvl <- anchor_boxs$lvl[kmean_model$cluster]
BOX_INFOS$seq <- anchor_boxs$seq[kmean_model$cluster]
  
# Save data

save(BOX_INFOS, file = revised_box_info_path)

anchor_boxs <- anchor_boxs[order(anchor_boxs$rank),]
rownames(anchor_boxs) <- 1:nrow(anchor_boxs)

save(anchor_boxs, file = anchor_boxs_path)
  
  
