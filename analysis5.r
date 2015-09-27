# Random forest with even less variable, to get rid of overfitting,
# and parallelization.
library(caret)
library(randomForest)
library(parallel)
library(doParallel)

set.seed(1234)

df <- read.csv('data/pml-training.csv', na.strings=c("NA", "#DIV/0!"))

inTrain <- createDataPartition(y=df$classe, p=.6, list=F)
training <- df[inTrain, ]
validation <- df[-inTrain, ]
rm("df")
rm("inTrain")

training <- training[, !grepl("X", names(training))]
training <- training[, !grepl("user_name", names(training))]
nzv <- nearZeroVar(training)
training <- training[, -nzv]

nas <- apply(training, 2, function(x) sum(is.na(x)) > 0)
training <- training[, !nas]

# The first four features seems to not be measurements.
training <- training[, -(1:4)]

system.time({
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
fit1 <- foreach(ntree=c(400, 400, 400), .combine=combine, .packages='randomForest') %dopar% {
    randomForest(classe ~ ., data=training, ntree=ntree, importance=T)
}
stopCluster(cl)
})

prediction <- predict(fit1, validation, type="class")



# df <- read.csv('data/pml-testing.csv', na.strings=c("NA", "#DIV/0!"))
# prediction <- predict(fit1, df, type="class")

# Il y a 52 features, donc 52 * (52 - 1) / 2 unique correlations.
library(corrplot)
corrplot(cor(training[, -53]),
         method="circle", type="lower", title="Correlation", mar=c(0,0,4,0))


varImpPlot(fit1, main="Most Important Features", n.var=12)


plot(df$roll_belt, jitter(as.numeric(df$classe), 2), pch=15, col=df$classe, cex=.5)


barplot(table(training$classe), main="Classes", col=rainbow(5))


# Notez qu'avec seulement les 9 features les plus importantes, l'accuracy est
# toujours excellente.
t2 <- training[, c(1, 2, 3, 39, 38, 41, 19, 14, 40, 53)]
system.time({
cl <- makeCluster(3)
registerDoParallel(cl)
fit2 <- foreach(ntree=c(400, 400, 400), .combine=combine, .packages='randomForest') %dopar% {
    randomForest(classe ~ ., data=t2, ntree=ntree, importance=T)
}
stopCluster(cl)
})
prediction2 <- predict(fit2, validation, type="class")
confusionMatrix(prediction, validation$classe)$table
confusionMatrix(prediction2, validation$classe)$table
barplot(table(prediction == prediction2))
