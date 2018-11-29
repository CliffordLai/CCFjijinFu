rm(list=ls())
invisible(gc())

if (!require("pacman")) install.packages("pacman")
pacman::p_load(data.table)


#----------------- Compute the local correlations matrices ---------------------------#
#Input path
setwd('G:/CCFjijinFu/')
train_Return_path <- "dataset/train_fund_return.csv"
test_Return_path <- "dataset/test_fund_return.csv"

#Read the return information
train_Return <- fread(train_Return_path, sep=",", header = TRUE,
                      na.strings="",
                      showProgress = TRUE)
train_Return <- as.data.frame(train_Return)
invisible(gc())
test_Return <- fread(test_Return_path, sep=",", header = TRUE,
                     na.strings="",
                     showProgress = TRUE)
test_Return <- as.data.frame(test_Return)
invisible(gc())

#Combine
fund_Return <- cbind(train_Return[,-1],test_Return[,-1])
dim(fund_Return)

rm(train_Return,test_Return)
invisible(gc())

#Input:
# Return: return data
# L: The length of pastvalues used to compute local correlations among p fund returns 
#Output: A list of two components
# corAB: A 3-d array containing d-L local p by p fund return correlation matrices
# dname: date of each local correlation matrix
windowFun <- function(Return,L){
  p <- nrow(Return)
  d <- ncol(Return)
  dname <- colnames(Return)
  
  VecMat <- array(0,dim = c(d-L,p,p) )
  for(j in (L+1):d){
    VecMat[j-L,,] <- cor(t(Return[,(j-L):j]))
  }
  
  VecMat[is.na(VecMat)] <- 0 #NA might happen because of constant sequence
  return(list(corAB=VecMat,dname=dname[(L+1):d]))
}

#Compute local correlation matrices with different lengths
lagSeq <- c(11,16,21) #There are many options other than this
length(lagSeq)
corFund <- list()
for(i in 1:length(lagSeq)){
  corFund[[i]] <- windowFun(fund_Return,lagSeq[i]-1)
  invisible(gc())
}
rm(fund_Return)
invisible(gc())
#----------------------------------------------------------------------------#





#------------------ Construct the data matrix X------------------------------#

lagLength <- length(lagSeq)

# Start time
(startDate <- corFund[[lagLength]]$dname[1])

# Record the initial date for local correlations with different lags
L <- numeric(lagLength-1)
for(l in 1:(lagLength-1)){
  L[l] <- which(corFund[[l]]$dname==corFund[[lagLength]]$dname[1])-1
}

dnames <- as.Date(corFund[[lagLength]]$dname) #date
(D <- length(dnames)) #The number of days
invisible(gc())

# This function rearrange the local correlation matrices for each fund pair and each day
doFeature <- function(i0,i1){
  (n <- (i1-i0+1)*(723*2-i0-i1)/2)   
  X1 <- matrix(0,n*D, 2+lagLength )
  invisible(gc())
  colnames(X1) <- c("i","j",paste("corFund_",lagSeq,sep=""))
  invisible(gc())
  rowDate1 <- integer(n*D)
  class(rowDate1) <- "Date"
  s <- 1
  
  for(i in i0:i1){
    K <- (723-i)*D
    #Pair index
    rowDate1[s:(s+K-1)] <- rep(dnames,each=723-i)
    X1[s:(s+K-1),1] <- i
    X1[s:(s+K-1),2] <- rep((i+1):723,D)
    invisible(gc())
    ic <- 2
    
    #corFund
    for(l in 1:(lagLength-1)){
      X1[s:(s+K-1),ic+l] <- matrix(t(corFund[[l]]$corAB[-c(1:L[l]),i,(i+1):723]),ncol=1)
    }
    X1[s:(s+K-1),ic+lagLength] <- matrix(t(corFund[[lagLength]]$corAB[,i,(i+1):723]),ncol=1)
    invisible(gc())
    ic <- ic+lagLength
    
    #Next fund
    s <- s+K
    cat(i,"")
  }
  
  return(list(rowDate=rowDate1,
              X=X1))
}

# The following procedure can be done once if memory is sufficiently large.
# Here it fits my computer with 16GB memory.
#1
i0 <- 1
i1 <- 155  #722
DX <- doFeature(i0,i1)
saveRDS(DX,"FEout/DX1.rds")
rm(DX) # Clean for more memory
invisible(gc())

#2
i0 <- 156
i1 <- 300  #722
DX <- doFeature(i0,i1)
saveRDS(DX,"FEout/DX2.rds")
rm(DX) # Clean for more memory
invisible(gc())

#3
i0 <- 301
i1 <- 722  #722
DX <- doFeature(i0,i1)
saveRDS(DX,"FEout/DX3.rds")
rm(DX) # Clean for more memory
invisible(gc())

#Save full data set
rm(corFund)
invisible(gc())

DX1 <- readRDS("FEout/DX1.rds")
DX2 <- readRDS("FEout/DX2.rds")
rowDate <- c(DX1$rowDate,DX2$rowDate)
X <- rbind(DX1$X,DX2$X)
rm(DX1,DX2)
invisible(gc())

#
for(j in 3){
  cat(j,"")
  DX_add <- readRDS(paste("FEout/","DX",j,".rds",sep=""))
  rowDate <- c(rowDate,DX_add$rowDate)
  X <- rbind(X,DX_add$X)
  rm(DX_add)
  invisible(gc())
}

# Save the data matrix X
saveRDS(rowDate,"rowDate.rds")
saveRDS(X,"FEout/X.rds")
#-----------------------------------------------------------------------------#





#------------------ Construct the response vector Y---------------------------#
# Clean for more memory
rm(DX,X,rowDate,corFund)
invisible(gc())

train_Cor_path <- "dataset/train_correlation.csv"
test_Cor_path <- "dataset/test_correlation.csv"

train_Cor <- fread(train_Cor_path, sep=",", header = TRUE,
                   na.strings="",
                   showProgress = TRUE)
train_Cor <- as.data.frame(train_Cor)
invisible(gc())

test_Cor <- fread(test_Cor_path, sep=",", header = TRUE,
                  na.strings="",
                  showProgress = TRUE)
test_Cor <- as.data.frame(test_Cor)
invisible(gc())

#Combine response
pairName <- test_Cor[,1]
All_Cor <- cbind(train_Cor[,-1],test_Cor[,-1])
All_Cor <- as.matrix(All_Cor)
rm(train_Cor,test_Cor)
invisible(gc())


#Rearrange the response to adapt to X
(rmD <- which(colnames(All_Cor)==startDate)) # the position of the last date before the start
(D2 <- ncol(All_Cor[,-c(1:rmD)]))
invisible(gc())
Y <- matrix(0, ncol=1, nrow=261003*D)
s <- 1
s0 <- 1
tc <- proc.time()
for(i in 1:722){
  K0 <- 723-i
  K <- (723-i)*D
  
  Y[s:(s+K-1)] <- as.vector(cbind(All_Cor[s0:(s0+K0-1),-c(1:rmD),drop=FALSE],matrix(0,K0,D-D2)))
  #Next fund
  s <- s+K
  s0 <- s0+K0
  cat(i,"")
  if(i%%10==0){invisible(gc())}
}
proc.time()-tc 

#Save the response vector Y
saveRDS(Y,"FEout/Y.rds")
#-----------------------------------------------------------------------------#



