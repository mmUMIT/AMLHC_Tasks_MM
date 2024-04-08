#import
data <- read.csv("AMLHC_Tasks_MM/food.csv", row.names = 1)
#
data_scale <- scale(data)
pca <- princomp(data_scale)
data_red <- pca$scores[,1:2]
data_red <- data_red * -1
plot(data_red[,"Comp.1"], data_red[,"Comp.2"])
ggplot2::qplot(data_red[,"Comp.1"], data_red[,"Comp.2"])
