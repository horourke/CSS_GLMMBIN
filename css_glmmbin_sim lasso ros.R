# Resampling Methods for Imbalanced Outcomes When Using Cluster-Sensitive Cross-Validation in Machine Learning
# O'Rourke, Xue, & Hilley 2026

library(tictoc)
library(parallel)
library(dplyr)
library(glmmLasso)
library(cv.glmmLasso)
ls(getNamespace("cv.glmmLasso"))
library(pROC)
library(PRROC)

dir.create("/rhome/hollyo/shared/dir_css_glmmbin")
save_dir <- ("/rhome/hollyo/shared/dir_css_glmmbin")

### loop through design to select training data for each condition using table 1 for # of clusters
tic()

# RNGkind makes child streams reproducible for parallel computing
RNGkind("L'Ecuyer-CMRG")
# seed for data-level reproduction
set.seed(0111287)

################################
###### 1. DATA GENERATION ######
################################

# !!! TODO change for each script
algo <- "lasso"
method <- "ros"

reps <- 200

j_vals <- c(10000, 3000, 4000, 1000) # number of clusters for n=60,000 (creating ground truth)
i_vals <- c(6, 20, 15, 60) # cluster size (intact for population data generation)
int_vals <- c(-3, -1, 0) # individual-level imbalance controlled by the intercept
tau_vals <- c(0,0.4,2)
p <- 20

design <- expand.grid(
  tau = tau_vals,
  int = int_vals,
  idx = 1:length(j_vals),
  rep = 1:reps
)

design$j <- j_vals[design$idx]
design$i <- i_vals[design$idx]
design$n <- design$j * design$i

# create cluster variable for modeling data split in design matrix
design$j_model <- ifelse(design$i == 6, 20,
                         ifelse(design$i == 20, 6,
                                ifelse(design$i == 15, 60,
                                       ifelse(design$i == 60, 15, NA))))

# create cluster variable for test data split in design matrix
design$j_test <- ifelse(design$i == 6, 5,
                        ifelse(design$i == 20, 1,
                               ifelse(design$i == 15, 15,
                                      ifelse(design$i == 60, 5, NA))))
design$cond <- with(
  design,
  as.integer(interaction(tau, int, j_model, i, drop = TRUE))
)

design$algo <- algo
design$method <- method
design$idx <- NULL

design <- design[order(design$cond, design$rep), ]
row.names(design) <- NULL

# re-order design matrix for simulating conditions by j/i, int, tau, rep
design <- design[order(design$cond, design$int, design$tau, design$rep), ]
row.names(design) <- NULL

# create function to loop through conditions
simulate_condition <- function(algo, method,
                               rep, cond, j, i,
                               beta0, tau,
                               j_model, j_test) {
  # Create data frame
  dat <- expand.grid(
   ind = 1:i,
   clust = 1:j
  )
  dat$algo <- algo
  dat$method <- method
  dat$rep <- rep
  dat$cond <- cond
  dat$j <- j
  dat$i <- i
  dat$n <- j * i
  dat$b0 <- beta0
  dat$tau <- tau
  dat$j_model <- j_model
  dat$j_test <- j_test
  
  # Cluster random effects
  u <- rnorm(j, mean = 0, sd = tau)
  
  # assign each person their cluster effect
  dat$u <- u[dat$clust]
  
  # create coefficient vector for intercept and predictors 
  beta = c(beta0,rep(.3,p*.75),rep(0,p*.25))
  
  # create n x 21 matrix to store data
  X <- cbind(1, matrix(rnorm(i*j*p), nrow = i*j))
  colnames(X) <- c("x_b0", paste0("x", 1:p))
  
  # linear y
  dat$lin <- as.vector(X %*% beta) + dat$u
  
  # convert to a probability
  dat$p <- plogis(dat$lin)
  
  # binary outcome
  dat$y <- rbinom(
    n = nrow(dat),
    size = 1,
    prob = dat$p
  )
  
  # save predictors & outcome in same dataset
  dat <- cbind(dat, X)
  dat
}

# function for fixing the errors coming from chosen lambdas issue
fit_path <- function(fix, rnd, data, family, lambdas) {
  lambdas <- sort(lambdas, decreasing = TRUE)   # must be high -> low
  out <- vector("list", length(lambdas))
  D <- NULL; Q <- NULL
  
  for (l in seq_along(lambdas)) {
    ctrl <- list(center = TRUE, standardize = TRUE)
    if (!is.null(D)) { ctrl$start <- D; ctrl$q_start <- Q }
    
    fit <- tryCatch(
      glmmLasso::glmmLasso(fix = fix, rnd = rnd, data = data,
                           family = family, lambda = lambdas[l],
                           control = ctrl),
      error = function(e) NULL
    )
    
    if (is.null(fit)) { D <- NULL; Q <- NULL; next }
    
    fit$lambda <- lambdas[l]
    out[[l]] <- fit
    
    d <- fit$Deltamatrix[fit$conv.step, ]
    if (all(is.finite(d))) {                     # only chain a clean solution
      D <- d
      Q <- fit$Q_long[[fit$conv.step + 1]]
      if (nrow(as.matrix(Q)) == 1) Q <- c(Q)
    } else { D <- NULL; Q <- NULL }              # otherwise cold-start next one
  }
  out
}

predict_path <- function(fits, newdata) {
  ok <- !vapply(fits, is.null, logical(1))
  m <- matrix(NA_real_, nrow(newdata), length(fits))
  if (any(ok)) m[, ok] <- do.call(cbind, lapply(fits[ok], predict, newdata = newdata))
  m
}


# helper function to create results matrix with the same # of rows as the design matrix
na_result <- function(r, status) {
  data.frame(
    algo       = design$algo[r],
    method     = design$method[r],
    rep        = design$rep[r],
    cond       = design$cond[r],
    tau        = design$tau[r],
    int        = design$int[r],
    j          = design$j_model[r],
    i          = design$i[r],
    status     = status,
    lambda     = NA_real_,
    aic        = NA_real_,
    conv_step  = NA_real_,
    loss_final = NA_real_,
    auc        = NA_real_,
    auprc      = NA_real_,
    brier      = NA_real_,
    loss_true  = NA_real_,
    auc_true   = NA_real_,
    auprc_true = NA_real_,
    brier_true = NA_real_,
    stringsAsFactors = FALSE
  )
}
############################################
###### 2. CLUSTER-SENSITIVE SPLITTING ######
############################################

# create function to repeat for each replication dataset
res <- mclapply(seq_len(nrow(design)), function(r){
  tryCatch({
#results <- lapply(seq_along(datasets), function(r) {
#!!!  
  # r <- 1

  # call function for data generation
  # this is inside the replication loop to reduce memory for parallelizing
    
    dat <- simulate_condition(
      algo = design$algo[r],
      method = design$method[r],
      rep = design$rep[r],
      cond = design$cond[r],
      j = design$j[r],
      i = design$i[r],
      beta0 = design$int[r],
      tau = design$tau[r],
      j_model = design$j_model[r],
      j_test = design$j_test[r]
    )

### 2a. cluster-sensitive splitting "ground truth" and modeling data ###

# select clusters for modeling data using j_model, the rest are ground truth
modeling_clusters <- sample(
  unique(dat$clust),
  size = dat$j_model[1]
)

# create modeling_data
modeling_data <- dat[
  dat$clust %in% modeling_clusters,
]

# create ground truth data
ground_truth <- dat[
  !dat$clust %in% modeling_clusters,
]
ground_truth$clust <- as.factor(ground_truth$clust)

### 2b. cluster-sensitive splitting of modeling_data into training and test data ###
test_clusters <- sample(
  modeling_clusters,
  size = dat$j_test[1]
)

# create test_data
test_data1 <- modeling_data[
  modeling_data$clust %in% test_clusters,
]

# create train_data
train_data1 <- modeling_data[
  !modeling_data$clust %in% test_clusters,
]

# convert cluster to factor in training and test data
train_data1$clust <- as.factor(train_data1$clust)
test_data1$clust <- as.factor(test_data1$clust)

#################################################################################
###### 3. CLUSTER-SENSITIVE K-FOLD CROSS-VALIDATION ON TRAINING DATA FOLDS ######
#################################################################################

### 3a. set up the cluster-sensitive k-fold CV code

# make Y a factor
train_data1$y <- as.factor(train_data1$y)

# define clusters
clusters <- unique(train_data1$clust)

# define 5 folds
folds <- sample(rep(1:5, length.out = length(clusters)))

# assign folds
fold_assign <- data.frame(
  clust = clusters,
  fold = folds
)

# initiate storage of lambdas, multi-lambda GLMM lasso, and loss vectors
loss_list <- vector(mode = 'list', length = 5)

# loop through folds
for(i in 1:5){
  
  # cluster assigned to validation fold in a given loop
  validation_clusters <- fold_assign$clust[
    fold_assign$fold == i
  ]
  
  # split data by cluster
  validation_data <- train_data1[
    train_data1$clust %in% validation_clusters,
  ]
  
  train_data_k <- train_data1[
    !train_data1$clust %in% validation_clusters,
  ]

### 3b. cluster-sensitive random oversampling (ROS) for the k-fold CV

  # create empty data frame for oversampled data
    train_data_oversampled <- data.frame()
    
  # object w/ training data clusters for fold in a given loop 
    training_clusters <- unique(train_data_k$clust)
  
    # loop through training clusters within this fold
  for (j in training_clusters) {
    clust_data <- train_data_k[train_data_k$clust == j, ]
    
    cases_0 <- clust_data[clust_data$y == 0, ]
    cases_1 <- clust_data[clust_data$y == 1, ]
    
    # oversample only if both classes exist
    if (nrow(cases_0) > 0 && nrow(cases_1) > 0) {
      # oversample only if there are more 0 cases than 1s
      if(nrow(cases_0) >  nrow(cases_1)) {
        oversampled_cases <- cases_1[
          sample(nrow(cases_1), nrow(cases_0), replace = TRUE),
        ]
        balanced_data <- rbind(cases_0, oversampled_cases)
      } else if (nrow(cases_0) <  nrow(cases_1)) {
        oversampled_cases <- cases_0[
          sample(nrow(cases_0), nrow(cases_1), replace = TRUE),
        ]
        balanced_data <- rbind(cases_1, oversampled_cases)
      } else {
        balanced_data <- clust_data
      }
    } else {
      balanced_data <- clust_data
    }
    train_data_oversampled <- rbind(train_data_oversampled, balanced_data)
  } # end of cluster loop
    
### 3c. hyperparameter tuning for glmmlasso for each fold of the k-fold cross-validation
  
  # create covariate list
  covs <- paste0("x", 1:20)
  
  # convert y to numeric in this fold for both training and validation data
  train_data_oversampled$y <- as.numeric(as.character(train_data_oversampled$y))
  validation_data$y <- as.numeric(as.character(validation_data$y))
  
  # convert cluster variable to factor in this fold
  train_data_oversampled$clust <- as.factor(train_data_oversampled$clust)
  
  # create formula
  fix <- reformulate(covs, response = "y")
  
  # create vector of lambda values
  if (i == 1) {
    lambdas <- cv.glmmLasso:::buildLambdas(fix = fix, rnd = list(clust = ~1),
                                           data = train_data_oversampled, nlambdas = 10)
    lambdas <- sort(lambdas, decreasing = TRUE)
  }
  
  # do glmmLasso for all lamba values in lambda
  modList_fold <- fit_path(fix = fix, rnd = list(clust = ~1),
                           data = train_data_oversampled,
                           family = binomial(link = "logit"),
                           lambdas = lambdas)

  # get response variable name
  response_var <- fix[[2]] %>% as.character()
  
  # pull y from validation data
    y_validation <- validation_data %>% 
      dplyr::pull(response_var)
  
  # predicting values for each of the glmmLasso model (10 lambdas) 
    predictionMatrix <- predict_path(modList_fold, validation_data)
  
  # employing the loss function in form loss(actual,predicted)
  # using loss function, calculating a list of loss values for each lambda
  loss <- function(actual, predicted)  {
    score <- -(actual * log(predicted) + (1 - actual) * log(1 -predicted))
    score[actual == predicted] <- 0
    score[is.nan(score)] <- Inf
    return(colMeans(score))
  }

  # create a vector of loss values for all lambdas in each fold
  loss_list[[i]] <- loss(actual = y_validation, predicted = predictionMatrix)
  # loss_list <- loss(actual = y_validation, predicted = predictionMatrix)
  
  # return NA for loss if model fails to converge
  # chosen lambdas are sometimes too small
  bad <- vapply(modList_fold, function(f)
    is.null(f) || f$conv.step >= 1000,
    logical(1))
  loss_list[[i]][bad] <- NA
  
  # free memory before next fold
  predictionMatrix <- NULL
  modList_fold <- NULL
  gc()
  
} ### end of k-fold loop

# free more memory (for parallel computing)
train_data_oversampled <- NULL
validation_data <- NULL
train_data_k <- NULL
convstep <- NULL
gc()

#######################################################################
###### 4. MINIMUM LAMBDA SELECTION & FULL TRAINING DATA ANALYSIS ######
#######################################################################

### 4a. minimum lambda selection

# create matrix of loss vectors
cvLossMatrix <- do.call(what = rbind, args = loss_list)

# average loss across 5 rows (folds) by columns (lambda values)
cvm <- colMeans(cvLossMatrix, na.rm = TRUE)

# chosen lambda corresponds to position of min avg loss across folds
if (all(is.na(cvm))) return(na_result(r, "cv_failed"))
chosenLambda <- lambdas[which.min(cvm)]
chosenLambda

### 4b. cluster-sensitive random oversampling in the full training dataset

# create empty data frame for oversampled data
train_data_overs_full <- data.frame()

# object w/ training data clusters for full training data
#NB: same as "clusters" above
training_clusters_full <- unique(train_data1$clust)

# loop through training clusters for full training data
for (j in training_clusters_full) {
  clust_data_full <- train_data1[train_data1$clust == j, ]
  
  cases_0_full <- clust_data_full[clust_data_full$y == 0, ]
  cases_1_full <- clust_data_full[clust_data_full$y == 1, ]
  
  # oversample only if both classes exist
  if (nrow(cases_0_full) > 0 && nrow(cases_1_full) > 0) {
    # oversample only if there are more 0 cases than 1s
    if(nrow(cases_0_full) >  nrow(cases_1_full)) {
      oversampled_cases_full <- cases_1_full[
        sample(nrow(cases_1_full), nrow(cases_0_full), replace = TRUE),
      ]
      balanced_data_full <- rbind(cases_0_full, oversampled_cases_full)
    } else if (nrow(cases_0_full) <  nrow(cases_1_full)) {
      oversampled_cases_full <- cases_0_full[
        sample(nrow(cases_0_full), nrow(cases_1_full), replace = TRUE),
      ]
      balanced_data_full <- rbind(cases_1_full, oversampled_cases_full)
    } else {
      balanced_data_full <- clust_data_full
    }
  } else {
    balanced_data_full <- clust_data_full
  }
  train_data_overs_full <- rbind(train_data_overs_full, balanced_data_full)
} # end of cluster loop

# convert y to numeric and clust to factor in final training data
train_data_overs_full$y <- as.numeric(as.character(train_data_overs_full$y))
train_data_overs_full$clust <- as.factor(train_data_overs_full$clust)

### 4c. glmmLasso on full oversampled training data with optimal lambda

# using function created to fix chosen lambdas issue
final_model <- fit_path(fix = fix, rnd = list(clust = ~1),
                        data = train_data_overs_full,
                        family = binomial(link = "logit"),
                        lambdas = chosenLambda)
if (is.null(final_model[[1]])) return(na_result(r, "final_failed"))

### 4d. Final prediction on full test data
final_prediction <- predict_path(final_model, test_data1)
loss_final <- loss(actual = test_data1$y, predicted = final_prediction)

# ### 4e. Final prediction on ground truth data
pop_prediction  <- predict_path(final_model, ground_truth)
loss_final_pop <- loss(actual = ground_truth$y, predicted = pop_prediction)

###################################################################
###### 5. SAVING OUTPUT METRICS FOR SIMULATION DATA ANALYSIS ######
###################################################################
# fitted values from final analysis
y_fit <- final_model[[1]][["fitted.values"]]

# AUC for this sim
# set AUC to NA if only 1 level of response variable
if (length(unique(test_data1$y)) < 2) {
  auc <- NA
} else {
  auc <- pROC::auc(
    response = test_data1$y,
    predictor = as.vector(final_prediction)
  )
}

# AUC for truth population (method-specific)
# set AUC to NA if only 1 level of response variable
if (length(unique(ground_truth$y)) < 2) {
  auc_truth <- NA
} else {
  auc_truth <- pROC::auc(
    response = ground_truth$y,
    predictor = as.vector(pop_prediction)
  )
}

# AUPRC for this sim
if (length(unique(test_data1$y)) < 2) {
  auprc_value <- NA
} else {
  auprc <- PRROC::pr.curve(
    scores.class0 = final_prediction[test_data1$y == 1],
    scores.class1 = final_prediction[test_data1$y == 0],
    curve = FALSE
  )
  auprc_value <- auprc$auc.integral
}

# AUPRC for truth (method-dependent)
if (length(unique(ground_truth$y)) < 2) {
  auprc_val_truth <- NA
} else {
  auprc_truth <- PRROC::pr.curve(
    scores.class0 = pop_prediction[ground_truth$y == 1],
    scores.class1 = pop_prediction[ground_truth$y == 0],
    curve = FALSE
  )
  auprc_val_truth <- auprc_truth$auc.integral
}

# Brier score for this sim
brier <- mean((test_data1$y - final_prediction)^2)

# Brier score for truth (method-dependent)
brier_truth <- mean((ground_truth$y - pop_prediction)^2)

# remove data to free up memory
rm(dat)
gc()

return(data.frame(
  algo = design$algo[r],
  method = design$method[r],
  rep = design$rep[r],
  cond = design$cond[r],
  tau = design$tau[r],
  int = design$int[r],
  j = design$j_model[r],
  i = design$i[r],
  status = "ok",
  lambda = final_model[[1]][["lambda"]],
  aic = final_model[[1]][["aic"]],
  conv_step = as.numeric(final_model[[1]][["conv.step"]]),
  loss_final = loss_final[1],
  auc = as.numeric(auc),
  auprc = auprc_value,
  brier = brier,
  loss_true = loss_final_pop[1],
  auc_true = as.numeric(auc_truth),
  auprc_true = auprc_val_truth,
  brier_true = brier_truth
))
}, error = function(e) na_result(r, paste0("error: ", conditionMessage(e))))
}, # end results loop
mc.cores = 12,
mc.set.seed = TRUE
)
ok <- vapply(res, is.data.frame, logical(1))
results <- do.call(rbind, res[ok])

# warning guard for if child streams OOM in parallel computign
if (any(!ok)) {
  warning(sprintf("%d of %d replications lost (worker died)", sum(!ok), length(ok)))
  results <- rbind(results,
                   do.call(rbind, lapply(which(!ok), na_result, status = "worker_died")))
}

results <- results[order(results$cond, results$rep), ]
rownames(results) <- NULL

toc()

# save results as .RDS
saveRDS(
  results,
  file = file.path(save_dir, "results_lasso_ros.RDS")
)

# save results as .csv
write.csv(results, paste0(save_dir,"/results_lasso_ros.csv"), row.names = FALSE)