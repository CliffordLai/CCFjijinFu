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

# Set the training dates and the validation day 
#  Here I used the last day (one day) to validate
#  The training set contains dates 61 days before the validation day
#  The 61 days in between is discarded.
uniqueRowDate <- unique(rowDate)
head(uniqueRowDate)
tail(uniqueRowDate)
valDate <- "2017-12-13"
(valaheadDate <- UrowDate[which(UrowDate==valDate)-61])
trainIndex <- (rowDate<valaheadDate) & (rowDate>=UrowDate[1])
valIndex <- (rowDate==valDate)
testIndex <- rowDate==as.Date("2018-03-16")
rm(UrowDate)

# Divide the full data set into training, validation, test.
train <- X[trainIndex,]
val <- X[valIndex,]
test <- X[testIndex,]
rm(X)
invisible(gc())

trainY <- Y[trainIndex]
valY <- Y[valIndex]
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

cat("Creating the 'validation' for modeling...\n")
valS = lgb.Dataset(data = val,label = valY,categorical_feature = catSet)
invisible(gc())


#######################################################
# Tune model
####################################################### 

#Output file names
suffix <- "1"
pathprefix <- "ModelLog/lgb/"
modelDir <- paste(pathprefix,"lgb_",suffix,".rds",sep="")
testOuput <- paste(pathprefix,"submitlgb_",suffix,".csv",sep="")
imp_path <- paste(pathprefix,"implgb_",suffix,".csv",sep="")

# Start recording the log file
#  used for choosing tuning parameters
logfile <- paste(pathprefix,"record_",suffix,".txt",sep="")
sink(logfile,split=TRUE) 

print("Modelling")
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
model <- lgb.train(params, trainS, valids = list(validation = valS), 
                   nthread = 4,
                   verbose= 1, record=TRUE,
                   eval = evalerror,
                   eval_freq = 10, early_stopping_rounds = 50)
tc <- proc.time()-tc
tc #
saveRDS.lgb.Booster(model,file = modelDir) 

# Evaluation
model$best_iter 
cat("Validation score @ best iter: ", 
    max(unlist(model$record_evals[["validation"]][["score"]][["eval"]])), 
    "\n\n")
plot(unlist(model$record_evals$validation$score$eval)) 

# Feature importance
cat("Feature importance: ")
impBooster <- lgb.importance(model, percentage = TRUE)
impBooster[,2:4] <- round(impBooster[,2:4],5)
print(impBooster)
fwrite(impBooster,imp_path)
invisible(gc())

#######################################################
# Make predictions on the test set
####################################################### 
#model <- readRDS.lgb.Booster(file = modelDir)
sub <- read.csv('dataset/submit_example.csv')

cat("Predictions: \n")
preds <- predict(model, data = test, num_iteration = model$best_iter )
sub$value <- preds
head(sub)
write.csv(x = sub, file = testOuput,
          quote = FALSE,row.names = FALSE)

# End recording 
sink() 


