# Plots csv's produced by testNetworkParams.py
 
library(ggplot2)
library(RColorBrewer)

log1 <- read.csv("trainingPerformanceLogistic.csv")
log2 <- read.csv("trainingPerformanceLogistic2.csv")
tan1 <- read.csv("trainingPerformanceTanh.csv")
tan2 <- read.csv("trainingPerformanceTanh2.csv")


#### logarithmic activation function
# reformat csv files into long orientation so I can use gglot
log <- merge(log1, log2, all = T)
dim(log)
names(log)

# Transpose matrix
log.t <- t(log)
n <- nrow(log.t)
head(log.t)
logMSEchars <- log.t[(5:n),]  
head(logMSEchars)

# Turn entries into numerics (coercing to matrix turned all entries into strings)
logMSE <- apply(logMSEchars, c(1,2), as.numeric)
head(logMSE)
tail(logMSE)

# Make dataframe
logMSE.df <- data.frame(logMSE)
head(logMSE.df)

# Format into long orientation (1 long column)
logMSE.df2 <- stack(logMSE.df, select = 1:15)
logMSE.df2[1:6,]
logMSE.df2[999:1005,]

# Check that it worked!
nrow(logMSE.df2)
logMSE.df2$x <- rep(c(1:1000), 15)
logMSE.df2$final <- rep(c(rep(0, 999), 1), 15)

# Pick some colors for plot
gs.pal <- colorRampPalette(c("firebrick","dodgerblue"), bias = 3, space = "rgb")
pal <- sample(gs.pal(15))

# Make separate subset for text
final <- logMSE.df2[which(logMSE.df2$final == 1),]

p <- ggplot(logMSE.df2, aes(x = x, y = values))
p + geom_line(aes(col = ind, group = ind)) + scale_color_manual(values = pal, breaks = c("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15"), name = "Model", labels = c(1:15)) + stat_summary(data = final, fun.y = "min", aes(group = ind, label = ind), geom = "text", size = 2.5, vjust = 1, hjust = -1) + theme(text = element_text(family = "mono")) + labs(x = "Iteration", y = "MSE", title = "MSE vs Number of Iterations, Logistic Activation Function", col = "Combination") 

# Check out which parameter combinations worked best
log[,c(1:4,1004)]
# Combo 9 is best, then 8 and 11
log[8,1:4] # 1000.0/(1000.0 + x) alpha/2      [24]     3.850372e-06
log[9,1:4] # 1000.0/(1000.0 + x) alpha/2      [24]     0.0001503116
log[11,1:4] # 1000.0/(1000.0 + x) alpha/2      [44]     9.061062e-06
# 8 and 9 were actually the same combination!

#### tanh activation function
tan <- merge(tan1, tan2, all = T)
dim(tan)

tan.t <- t(tan)
n <- nrow(tan.t)
head(tan.t)
tanMSEchars <- tan.t[(5:n),]  
head(tanMSEchars)

tanMSE <- apply(tanMSEchars, c(1,2), as.numeric)
head(tanMSE)
tail(tanMSE)

tanMSE.df <- data.frame(tanMSE)
head(tanMSE.df)

tanMSE.df2 <- stack(tanMSE.df, select = 1:19)
tanMSE.df2[1:6,]
tanMSE.df2[999:1005,]

nrow(tanMSE.df2)
tanMSE.df2$x <- rep(c(1:1000), 19)

pal <- sample(gs.pal(19))

# Dear lord.
p <- ggplot(tanMSE.df2, aes(x = x, y = values))
p + geom_line(aes(col = ind, group = ind)) + scale_color_manual(values = pal, breaks = c("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15", "X16", "X17", "X18", "X19"), name = "Model", labels = c(1:19)) + theme(text = element_text(family = "mono")) + labs(x = "Iteration", y = "MSE", title = "MSE vs Number of Iterations, TanH Activation Function", col = "Combination")










