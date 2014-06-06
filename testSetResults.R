results <- read.csv("results_final.csv")

str(results)
names(results)

# Add column for trial number
results$Trial <- sort(rep(1:20, 5))
n <- nrow(results)

attach(results)

mean(Movement == MaxPred) # 55% success rate

# Add column for 'success'
results$Success <- Movement == MaxPred
attach(results)


table(Success, Trial)


## Look at final MSE
is.factor(TrainingSetMSE) # Well that's dumb.
unique(TrainingSetMSE) # all pretty low
unique(TestSetMSE)
as.numeric(toString(TrainingSetMSE))