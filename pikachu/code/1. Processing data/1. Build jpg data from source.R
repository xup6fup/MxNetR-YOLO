
# Get data from url: 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/'

# Libraries

library(mxnet)
library(abind)
library(jpeg)
library(RCurl)

# Set path

data.path <- "pikachu/data/"

train.path <- "pikachu/data/train/"
train_label.path <- "pikachu/data/train_label.txt"

val.path <- "pikachu/data/val/"
val_label.path <- "pikachu/data/val_label.txt"

# Creat directories and download files

if (!dir.exists(data.path)) {dir.create(data.path, recursive = TRUE)}
if (!dir.exists(train.path)) {dir.create(train.path, recursive = TRUE)}
if (!dir.exists(val.path)) {dir.create(val.path, recursive = TRUE)}
if (!file.exists(paste0(data.path, 'train.rec'))) {
  download.file(url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/train.rec', destfile = paste0(data.path, 'train.rec'), method = "libcurl")
}
if (!file.exists(paste0(data.path, 'train.idx'))) {
  download.file(url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/train.idx', destfile = paste0(data.path, 'train.idx'), method = "libcurl")
}
if (!file.exists(paste0(data.path, 'val.rec'))) {
  download.file(url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/val.rec', destfile = paste0(data.path, 'val.rec'), method = "libcurl")
}

# Parameter

data_shape <- c(256, 256, 3)
batch_size <- 32

# Load data
# These data are in an iterator. We need to use this iterator to generate jpg file

train_iter <- mx.io.ImageDetRecordIter(path.imgrec = paste0(data.path, "train.rec"),
                                       batch.size = batch_size,
                                       data.shape = data_shape,
                                       shuffle = TRUE,
                                       mean = TRUE,
                                       rand.crop = 1,
                                       min.crop.object.coverages = 0.95) 

val_iter <- mx.io.ImageDetRecordIter(path.imgrec = paste0(data.path, "val.rec"),
                                     batch.size = batch_size,
                                     data.shape = data_shape,
                                     shuffle = FALSE,
                                     mean = TRUE)

# Training data into R files.

train_iter$reset()

Train_list <- list()

while (train_iter$iter.next()){
  Train_list[[length(Train_list)+1]] <- train_iter$value()
  if (sum(as.array(Train_list[[length(Train_list)]]$label)[9,]) > 0){
    stop('find none pikachu in data.')
  }
}

# Get data

Pikachu_train_data <- array(NA, dim = c(256, 256, 3, batch_size * length(Train_list)))

for (i in 1:length(Train_list)){
  Pikachu_train_data[,,,(1+batch_size*(i-1)):(batch_size*i)] <- as.array(Train_list[[i]]$data)
}

# Rotate data

for (i in 1:dim(Pikachu_train_data)[4]){
  Pikachu_train_data[,,1,i] <- t(Pikachu_train_data[,,1,i])
  Pikachu_train_data[,,2,i] <- t(Pikachu_train_data[,,2,i])
  Pikachu_train_data[,,3,i] <- t(Pikachu_train_data[,,3,i])
}

# Get Label

Pikachu_train_label_old <- array(NA, dim = c(4, batch_size * length(Train_list)))

for (i in 1:length(Train_list)){
  Pikachu_train_label_old[,(1+batch_size*(i-1)):(batch_size*i)] <- as.array(Train_list[[i]]$label)[10:13,]
}  

# Make order of label into col_left, col_right, row_bot, row_top

Pikachu_train_label <- array(NA, dim = c(4, batch_size * length(Train_list)))

Pikachu_train_label[1,] <- Pikachu_train_label_old[1,]
Pikachu_train_label[2,] <- Pikachu_train_label_old[3,]
Pikachu_train_label[3,] <- Pikachu_train_label_old[4,]
Pikachu_train_label[4,] <- Pikachu_train_label_old[2,]

#####################################################################

# Vaildation data into R files.

val_iter$reset()

val_list <- list()

while (val_iter$iter.next()){
  val_list[[length(val_list)+1]] <- val_iter$value()
  if (sum(as.array(val_list[[length(val_list)]]$label)[9,]) > 0){
    stop('find none pikachu in data.')
  }
}

# Get data

Pikachu_val_data <- array(NA, dim = c(256, 256, 3, batch_size * length(val_list)))

for (i in 1:length(val_list)){
  Pikachu_val_data[,,,(1+batch_size*(i-1)):(batch_size*i)] <- as.array(val_list[[i]]$data)
}

# Rotate data

for (i in 1:dim(Pikachu_val_data)[4]){
  Pikachu_val_data[,,1,i] <- t(Pikachu_val_data[,,1,i])
  Pikachu_val_data[,,2,i] <- t(Pikachu_val_data[,,2,i])
  Pikachu_val_data[,,3,i] <- t(Pikachu_val_data[,,3,i])
}

# Get Label

Pikachu_val_label_old <- array(NA, dim = c(4, batch_size * length(val_list)))

for (i in 1:length(val_list)){
  Pikachu_val_label_old[,(1+batch_size*(i-1)):(batch_size*i)] <- as.array(val_list[[i]]$label)[10:13,]
}  

# Make order of label into col_left, col_right, row_bot, row_top

Pikachu_val_label <- array(NA, dim = c(4, batch_size * length(val_list)))

Pikachu_val_label[1,] <- Pikachu_val_label_old[1,]
Pikachu_val_label[2,] <- Pikachu_val_label_old[3,]
Pikachu_val_label[3,] <- Pikachu_val_label_old[4,]
Pikachu_val_label[4,] <- Pikachu_val_label_old[2,]

#####################################################################

# Write out

for (i in 1:dim(Pikachu_train_data)[4]){
  writeJPEG(Pikachu_train_data[,,,i]/255, target = paste0(train.path, i,".jpeg"), quality = 1, bg = "white")
}

train_box_info <- data.frame(obj_name = 'pikachu',
                             col_left = Pikachu_train_label[1,],
                             col_right = Pikachu_train_label[2,],
                             row_bot = Pikachu_train_label[3,],
                             row_top = Pikachu_train_label[4,],
                             prob = 1,
                             img_id = 1:dim(Pikachu_train_data)[4],
                             stringsAsFactors = FALSE)

write.table(train_box_info, file = train_label.path, sep = "\t", row.names = FALSE)

for (i in 1:dim(Pikachu_val_data)[4]){
  writeJPEG(Pikachu_val_data[,,,i]/255, target = paste0(val.path, i,".jpeg"), quality = 1, bg = "white")
}

val_box_info <- data.frame(obj_name = 'pikachu',
                           col_left = Pikachu_val_label[1,],
                           col_right = Pikachu_val_label[2,],
                           row_bot = Pikachu_val_label[3,],
                           row_top = Pikachu_val_label[4,],
                           prob = 1,
                           img_id = 1:dim(Pikachu_val_data)[4],
                           stringsAsFactors = FALSE)

write.table(val_box_info, file = val_label.path, sep = "\t", row.names = FALSE)
