results <- read.csv("results_final.csv", stringsAsFactors = F)

str(results)
names(results)

####### Clean up data set
# Add column for trial number
results$Trial <- sort(rep(1:20, 5))

attach(results)

# Add column for 'success'
results$Success <- Movement == MaxPred

# Make some columns factors
Movement <- as.factor(Movement)
File <- as.factor(File)
MaxPred <- as.factor(MaxPred)

# Make some columns numeric
results[,c(4:8, 13:16)] <- apply(as.matrix(results[,c(4:8, 13:16)]), c(1,2), as.numeric)
attach(results)

####### Look at some results
mean(Success) # 55% success rate
table(Success, Trial)

## Look at final MSE
summary(TrainingSetMSE)
summary(TestSetMSE)
unique(TestSetMSE)


library(ggplot)
icons <- results[which(Movement == "Icons"), c(1,3:8)]
mod <- results[which(Movement == "Modernism"), c(1,3:8)]
crit <- results[which(Movement == "CritRealism"), c(1,3:8)]
soc <- results[which(Movement == "SocRealism"), c(1,3:8)]
a <- results[which(Movement == "Aivazovskii"), c(1,3:8)]

stack(icons, select = 1:15)

ggplot(results, aes(Movement, MaxPred, fill = z)) + geom_raster(hjust = 0, vjust = 0)


