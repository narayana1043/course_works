train <- read.csv('./Data/train.csv')
test <- read.csv('./Data/test.csv')

cor_finder <- function(use_column){
  print(use_column)
  # A basic linear regression with only few features
  # new_test = test[use_column]
  # plot(new_train)
  r <- cor(train[use_column], train$SalePrice)
  print(r)
  return (r)
}

index <- c(1:length(colnames(train)))
colnames <- colnames(train)
dtype <- sapply(train, class)

for (i in index){
  # print(i)
  # print(colnames[i])
  if (is.integer(train[colnames[i]]) | is.numeric(train[colnames[i]])){
    x<-cor_finder(colnames[i])
    print(x)
  }
}


