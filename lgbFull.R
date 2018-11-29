rm(list=ls())
invisible(gc())
rm(list=ls())
invisible(gc())
#load library
if (!require("pacman")) install.packages("pacman")
pacman::p_load(lightgbm,data.table)
set.seed(1)               
options(scipen = 9999, warn = -1, digits= 4)


#######################################################
# Read data
####################################################### 
#rm(list=ls())
setwd('G:/CCFjijinFu/')
rowDate <- readRDS("FEout/rowDate.rds")
X <- readRDS("FEout/X.rds")
Y <- readRDS("FEout/Y.rds")

# Divide the full data set into training and test.
trainIndex <- (rowDate<=as.Date("2017-12-13"))
testIndex <- rowDate==as.Date("2018-03-16")

train <- X[trainIndex,]
test <- X[testIndex,]
rm(X)
invisible(gc())

trainY <- Y[trainIndex]
rm(Y,trainIndex,valIndex,testIndex)
invisible(gc())

# The first two columns indicate the two funds of the correlation
catSet <- c("i","j")
ncol(train)
colnames(test)

# LGB data set
cat("Creating the 'train' for modeling...\n")
trainS = lgb.Dataset(data = train,label = trainY,categorical_feature = catSet)
invisible(gc())

#######################################################
# Tune model
####################################################### 

#Output file names
suffix <- "1"
pathprefix <- "ModelLog/lgb/"
modelDir <- paste(pathprefix,"lgbfull_",suffix,".rds",sep="")
testOuput <- paste(pathprefix,"submitlgbfull_",suffix,".csv",sep="")
imp_path <- paste(pathprefix,"implgbfull_",suffix,".csv",sep="")

# Start recording the log file
#  used for choosing tuning parameters
logfile <- paste(pathprefix,"recordfull_",suffix,".txt",sep="")
sink(logfile,split=TRUE) 

print("Modelling")

# Use the best parameters from lgbTrainVal
params = list(objective = "regression_l1",
              num_iterations=250,
              learning_rate=0.05,
              num_leaves= 17,
              min_child_samples= 300,
              max_bin= 55,
              max_depth=3,
              subsample= 0.65,  
              subsample_freq= 1,
              colsample_bytree= 0.4,  
              min_child_weight= 0,
              min_split_gain= 0.00001,
              #  lambda_l2=0.1,
              boosting="gbdt") # 
# Record parameters
print(params) 

# Custom score function
evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")

  MAE <- mean(abs(preds-labels))
  TMAPE <- mean(abs((preds-labels)/(1.5-labels)))
  score <- (2/(2+MAE+TMAPE))^2

  return(list(name = "score", value = score, higher_better = TRUE))
}

#######################################################
# Start training
####################################################### 
set.seed(193)
tc <- proc.time()
model <- lgb.train(params, trainS, valids = list(training = trainS), nthread = 4,
                   verbose= 1, record=TRUE,
                   eval = evalerror,
                   eval_freq = 10)
tc #
saveRDS.lgb.Booster(model,file = modelDir) 

# Evaluation on the training set
plot(unlist(model$record_evals$training$score$eval))

# Feature importance
cat("Feature importance: ")
impBooster <- lgb.importance(model, percentage = TRUE)
impBooster[,2:4] <- round(impBooster[,2:4],5)
fwrite(impBooster,imp_path)
invisible(gc())
impBooster

#######################################################
# Make predictions on the test set
####################################################### 
#model <- readRDS.lgb.Booster(file = modelDir)
sub <- read.csv('dataset/submit_example.csv')

cat("Predictions: \n")
nstep <- 145
preds <- predict(model, data = test, num_iteration = nstep )
sub$value <- preds
head(sub)
write.csv(x = sub, file = testOuput,
          quote = FALSE,row.names = FALSE)

# End recording 
sink() 


