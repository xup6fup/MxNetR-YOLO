
# This mirror website is contributed by Joseph Chet Redmon.
# https://pjreddie.com/projects/pascal-voc-dataset-mirror/

path_train <- 'http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar'
path_test <- 'http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar'

# Start to download

data_dict <- 'data/'

if (!dir.exists(data_dict)) {dir.create(data_dict)}

train_local_path <- paste0(data_dict, 'VOCtrainval_06-Nov-2007.tar')
test_local_path <- paste0(data_dict, 'VOCtest_06-Nov-2007.tar')

if (!file.exists(train_local_path)) {
  download.file(url = path_train, destfile = train_local_path, method = 'wget', quiet = TRUE)
}

if (!file.exists(test_local_path)) {
  download.file(url = path_test, destfile = test_local_path, method = 'wget', quiet = TRUE)
}

#untar(tarfile = train_local_path, exdir = data_dict)
