#import
data <- read.csv("AMLHC_Tasks_MM/food.csv", row.names = 1)


# 1 Load dataset
diabetes_data <- read.csv("AMLHC_Tasks_MM/diabetes.csv")

# 2 Define and apply a function for IQR based outlier detection only on numeric columns
detect_outliers_IQR <- function(x) {
  if(is.numeric(x)) {
    Q1 <- quantile(x, 0.25, na.rm = TRUE)
    Q3 <- quantile(x, 0.75, na.rm = TRUE)
    IQR <- Q3 - Q1
    x[x < (Q1 - 1.5 * IQR) | x > (Q3 + 1.5 * IQR)] <- NA
  }
  return(x)
}

# Apply the function only to numeric columns
diabetes_data_clean <- as.data.frame(lapply(diabetes_data, detect_outliers_IQR))
diabetes_data_clean

# 3 Replace outliers with NA and select only complete cases
diabetes_data_complete <- diabetes_data_clean[complete.cases(diabetes_data_clean), ]

# Assuming you have further steps that involve FSelector or plotting,
# Make sure to replace feature_high_IG and feature_low_IG with actual column names

#4  Load the Hmisc package
library(Hmisc)

# Apply the describe function to your dataset
describe_results <- describe(diabetes_data_complete)

# Print the results
print(describe_results)

#5
diabetes_data_complete <- diabetes_data_clean[complete.cases(diabetes_data_clean), ]
diabetes_data_complete

# Rank features using information gain (FSelector package needed)
# install.packages("FSelector") # Uncomment if not installed
# might also need
# install.packages("RWekajars")
# install.packages("RWeka") to install RWekajars
library(FSelector)
weights <- information.gain(class ~ ., diabetes_data_complete)
weights

# Create boxplot and distribution plots of one discriminating and one non-discriminating feature
par(mfrow=c(2,2)) # Setting up the plotting area for four plots

# Boxplots
boxplot(diabetes_data_complete$feature_high_IG, main="Boxplot of High IG Feature")
boxplot(diabetes_data_complete$feature_low_IG, main="Boxplot of Low IG Feature")

# Histograms (Distribution Plots)
hist(diabetes_data_complete$feature_high_IG, main="Distribution of High IG Feature", xlab="Feature High IG", breaks=20)
hist(diabetes_data_complete$feature_low_IG, main="Distribution of Low IG Feature", xlab="Feature Low IG", breaks=20)

