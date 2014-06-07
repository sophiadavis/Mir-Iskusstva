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
tab <- table(Trial, Success)
correct <- cbind(data.frame(tab)[21:40,c(1,3)], data.frame(tab)[1:20,3])
names(correct) <- c("Trial", "Correct", "Incorrect")
correct 

## Look at final MSE
summary(TrainingSetMSE)
summary(TestSetMSE)
unique(TestSetMSE)

correct2 <- merge(correct, data.frame(Trial, TrainingSetMSE), by = "Trial")
correct3 <- merge(correct2, data.frame(Trial, TestSetMSE), by = "Trial")
final <- unique(correct3)[order(unique(correct3$Trial)),]
row.names(final) <- NULL
final[,2:5]

library(ggplot)
icons <- results[which(Movement == "Icons"), c(1,3:8)]
mod <- results[which(Movement == "Modernism"), c(1,3:8)]
crit <- results[which(Movement == "CritRealism"), c(1,3:8)]
soc <- results[which(Movement == "SocRealism"), c(1,3:8)]
a <- results[which(Movement == "Aivazovskii"), c(1,3:8)]

iconPreds <- stack(icons, select = 3:7)
modPreds <- stack(mod, select = 3:7)
critPreds <- stack(crit, select = 3:7)
socPreds <- stack(soc, select = 3:7)
aPreds <- stack(a, select = 3:7)

ggplot(iconPreds, aes(x = ind, y = values, fill = ind)) + guides(fill = F) + stat_summary(fun.y = "mean", geom = "bar") + theme(text = element_text(family = "mono")) + labs(x = "Classification", y = "Average Output", title = "Icons: Average Predicted Values") + scale_x_discrete(breaks = c("Aivazovskii", "CritRealism", "Icons", "Modernism", "SocRealism"), labels = c("Aivazovskii", "Critical Realism", "Icons", "Modernism", "Socialist Realism"))

ggplot(modPreds, aes(x = ind, y = values, fill = ind)) + guides(fill = F) + stat_summary(fun.y = "mean", geom = "bar") + theme(text = element_text(family = "mono")) + labs(x = "Classification", y = "Average Output", title = "Modernism: Average Predicted Values") + scale_x_discrete(breaks = c("Aivazovskii", "CritRealism", "Icons", "Modernism", "SocRealism"), labels = c("Aivazovskii", "Critical Realism", "Icons", "Modernism", "Socialist Realism"))

ggplot(critPreds, aes(x = ind, y = values, fill = ind)) + guides(fill = F) + stat_summary(fun.y = "mean", geom = "bar") + theme(text = element_text(family = "mono")) + labs(x = "Classification", y = "Average Output", title = "Critical Realism: Average Predicted Values") + scale_x_discrete(breaks = c("Aivazovskii", "CritRealism", "Icons", "Modernism", "SocRealism"), labels = c("Aivazovskii", "Critical Realism", "Icons", "Modernism", "Socialist Realism"))

ggplot(socPreds, aes(x = ind, y = values, fill = ind)) + guides(fill = F) + stat_summary(fun.y = "mean", geom = "bar") + theme(text = element_text(family = "mono")) + labs(x = "Classification", y = "Average Output", title = "Socialist Realism: Average Predicted Values") + scale_x_discrete(breaks = c("Aivazovskii", "CritRealism", "Icons", "Modernism", "SocRealism"), labels = c("Aivazovskii", "Critical Realism", "Icons", "Modernism", "Socialist Realism"))

ggplot(aPreds, aes(x = ind, y = values, fill = ind)) + guides(fill = F) + stat_summary(fun.y = "mean", geom = "bar") + theme(text = element_text(family = "mono")) + labs(x = "Classification", y = "Average Output", title = "Aivazovskii: Average Predicted Values") + scale_x_discrete(breaks = c("Aivazovskii", "CritRealism", "Icons", "Modernism", "SocRealism"), labels = c("Aivazovskii", "Critical Realism", "Icons", "Modernism", "Socialist Realism"))


