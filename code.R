library(keras)
#install_keras()
######################################################
mnist <- dataset_mnist()

#for(i in 1:1) {
v = 10000 * 2
xtr <- mnist$train$x
ytr <- mnist$train$y
ytest <- mnist$test$y
xtr <- array_reshape(xtr, c(60000, 28, 28, 1))
xtr <- xtr/255

xtest <- array_reshape(mnist$test$x, c(nrow(mnist$test$x), 28, 28, 1)) / 255
nc <- 10
ytr <- to_categorical(ytr, nc)
ytest <- to_categorical(ytest, nc)

# ep = c(0, 0.1, 0.2, 0.3)
# p <- runif(784)
# xtr <- xtr + (p/sum(p) * ep[4])

model <- keras_model_sequential() %>%
  layer_conv_2d(filters=16, kernel_size=c(3,3), activation = 'relu', input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = nc, activation = 'softmax')

#compile
model %>% compile( optimizer = optimizer_adadelta(), loss= loss_categorical_crossentropy, metrics=c("accuracy"))

model %>%
  fit(xtr,ytr, batch_size = 128, epochs = 12, validation_split= 0.2) # training data

score <- model %>% evaluate(xtest, ytest)
cat('Test loss: ', score$loss, "\n")
cat('Test accuracy: ', score$acc, "\n")
  
#}




