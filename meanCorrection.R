setwd('G:/CCFjijinFu/')

#####################################################################
# Compute the overall mean and standard deviation for each pair
##################################################################### 
if (!require("pacman")) install.packages("pacman")
pacman::p_load(data.table)

train_Cor <- fread("dataset/train_correlation.csv", sep=",", header = TRUE,
                   na.strings="",
                   showProgress = TRUE)
train_Cor <- as.data.frame(train_Cor)
invisible(gc())

test_Cor <- fread("dataset/test_correlation.csv", sep=",", header = TRUE,
                  na.strings="",
                  showProgress = TRUE)
test_Cor <- as.data.frame(test_Cor)
invisible(gc())

pairName <- test_Cor[,1]
All_Cor <- cbind(train_Cor[,-1],test_Cor[,-1])
All_Cor <- as.matrix(All_Cor)
rm(train_Cor,test_Cor)
invisible(gc())
dim(All_Cor)

sdCor <- apply(All_Cor,1,sd)
meanCor <- apply(All_Cor,1,mean)


#####################################################################
# Compute the overall mean and standard deviation for each pair
##################################################################### 
filename <- 'ModelLog/lgb/sublgbfull1'
inputName <- paste(filename,".csv",sep="")
outputName <- paste(filename,"_CM.csv",sep="")

# Read the predictions from lgb
sub <- read.csv(inputName)

# Compare with the overall mean predictions
summary(sub$value-meanCor)

# Set the tolerance to replace lgb prediction with overall means
correctIndex <- abs(sub$value-meanCor)>0.05 & sdCor<0.025
sum(correctIndex)
sum(correctIndex)/nrow(sub)

# Apply mean correction
sub$value[correctIndex] <- meanCor[correctIndex]
write.csv(x = sub, file = outputName,
          quote = FALSE,row.names = FALSE)
head(sub)
