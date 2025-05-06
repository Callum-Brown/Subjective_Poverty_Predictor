# Load required libraries
install.packages("dplyr")
install.packages("zoo")
library(dplyr)
library(zoo)  # For rolling averages

# ===============================
# Load Data
# ===============================

# Import training data
edu_train <- read.csv("f-2024-kaggle-contest-for-classification/module_Education_train_set.csv")
hh_train <- read.csv("f-2024-kaggle-contest-for-classification/module_HouseholdInfo_train_set.csv")
sub_pov_train <- read.csv("f-2024-kaggle-contest-for-classification/module_SubjectivePoverty_train_set.csv")

# Import test data
edu_test  <- read.csv("f-2024-kaggle-contest-for-classification/module_Education_test_set.csv")
hh_test <- read.csv("f-2024-kaggle-contest-for-classification/module_HouseholdInfo_test_set.csv")

# ===============================
# Merge Datasets (Train)
# ===============================

# Create a unique ID
edu_train$psu_hh_idcode <- paste(edu_train$psu, edu_train$hh, edu_train$idcode, sep = "_")
hh_train$psu_hh_idcode  <- paste(hh_train$psu, hh_train$hh, hh_train$idcode, sep = "_")

# Merge on ID
merged_data <- merge(edu_train, hh_train, by = "psu_hh_idcode")
merged_data <- merge(merged_data, sub_pov_train, by = "psu_hh_idcode", all.x = TRUE)

# Preview merged training data
glimpse(merged_data)
cat("Training rows:", nrow(merged_data), "\n")

# Create ordinal target variable
merged_data$Sub_Pov_Ranking <- apply(merged_data[, grep("subjective_poverty_", names(merged_data))], 1, function(row) {
  which(row == 1)
})

# ===============================
# Merge Datasets (Test)
# ===============================

edu_test$psu_hh_idcode <- paste(edu_test$psu, edu_test$hh, edu_test$idcode, sep = "_")
hh_test$psu_hh_idcode  <- paste(hh_test$psu, hh_test$hh, hh_test$idcode, sep = "_")
merged_data_test <- merge(edu_test, hh_test, by = "psu_hh_idcode")

# Add placeholder columns for missing subjective poverty data
for (i in 1:10) {
  merged_data_test[[paste0("subjective_poverty_", i)]] <- NA
}
merged_data_test$Sub_Pov_Ranking <- NA

# ===============================
# Combine Train and Test
# ===============================

merged_data_comb <- rbind(merged_data, merged_data_test)
glimpse(merged_data_comb)

# Add row index (HH)
merged_data_comb$HH <- 1:nrow(merged_data_comb)

# ===============================
# Rolling Averages & Standard Deviations
# ===============================

# Custom function for rolling mean
rolling_av <- function(x, n) {
  rollapply(x, width = 2*n + 1, FUN = mean, fill = NA, align = "center", na.rm = TRUE)
}

# Custom function for rolling SD
rolling_sd <- function(x, n) {
  rollapply(x, width = 2*n + 1, FUN = sd, fill = NA, align = "center", na.rm = TRUE)
}

# Apply to Sub_Pov_Ranking
AV_means <- rolling_av(merged_data_comb$Sub_Pov_Ranking, 10)
SD_sds   <- rolling_sd(merged_data_comb$Sub_Pov_Ranking, 10)

# ===============================
# Model Training Prep
# ===============================

# Prepare feature matrix (remove target column)
X <- merged_data_comb[, !names(merged_data_comb) %in% c("Sub_Pov_Ranking")]
X[is.na(X)] <- -1  # Replace NAs with -1
X <- as.matrix(data.matrix(X))  # Ensure numeric matrix

# Target
y <- merged_data_comb$Sub_Pov_Ranking

# Create XGBoost DMatrix
dtrain <- xgb.DMatrix(data = X, label = y)

# Model Parameters
params <- list(
  objective = "multi:softprob",
  eval_metric = "mlogloss",
  num_class = 10
)

# ===============================
# Cross-validation
# ===============================

cv_results <- xgb.cv(
  params = params,
  data = dtrain,
  nfold = 7,
  nrounds = 100,
  early_stopping_rounds = 10,
  verbose = TRUE
)