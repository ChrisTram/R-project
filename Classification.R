# install.packages("rpart")
# install.packages("randomForest")
# install.packages("kknn")
# install.packages("ROCR")
# install.packages("e1071")
# install.packages("nnet")

library.path <- .libPaths()
library("rpart", lib.loc = library.path)
library("randomForest", lib.loc = library.path)
library("kknn", lib.loc = library.path)
library("e1071", lib.loc = library.path)
library("nnet", lib.loc = library.path)
library("ROCR", lib.loc = library.path)

#-------------------------#
# PREPARATION DES DONNEES #
#-------------------------#

# Chargement des donnees
datas <- read.csv("C:\\Users\\maelg\\OneDrive\\Bureau\\R-project-main\\Data Projet.csv", header = TRUE, sep = ",", dec = ".", stringsAsFactors = T) 
datas_new <- read.csv("C:\\Users\\maelg\\OneDrive\\Bureau\\R-project-main\\Data Projet New.csv", header = TRUE, sep = ",", dec = ".", stringsAsFactors = T) 

# On remplace "Oui" par 1 et "Non" par 0 pour la colonne RESPONSE
# datas$RESPONSE <- ifelse(datas$RESPONSE=='Oui', 1, ifelse(datas$RESPONSE=='Non', 0, 2))


# Creation des ensembles d'apprentissage et de test
datas_EA <- datas[1:5400,]
datas_ET <- datas[5401:6400,]


#-------------------------#
# ARBRE DE DECISION RPART #
#-------------------------#

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC
test_rpart <- function(arg1, arg2, arg3, arg4){
  # Apprentissage du classifeur
  dt <- rpart(RESPONSE~., datas_EA, parms = list(split = arg1), control = rpart.control(minbucket = arg2))

  # Tests du classifieur : classe predite
  dt_class <- predict(dt, datas_ET, type="class")

  # Matrice de confusion
  print(table(datas_ET$RESPONSE, dt_class))

  # Tests du classifieur : probabilites pour chaque prediction
  dt_prob <- predict(dt, datas_ET, type="prob")

  # Courbes ROC
  dt_pred <- prediction(dt_prob[,2], datas_ET$RESPONSE)
  dt_perf <- performance(dt_pred,"tpr","fpr")
  plot(dt_perf, main = "Arbres de décision rpart()", add = arg3, col = arg4)

  # Calcul de l'AUC et affichage par la fonction cat()
  dt_auc <- performance(dt_pred, "auc")
  cat("AUC = ", as.character(attr(dt_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()

  return (dt)
}

#----------------#
# RANDOM FORESTS #
#----------------#

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC
test_rf <- function(arg1, arg2, arg3, arg4){
  # Apprentissage du classifeur
  rf <- randomForest(RESPONSE~., datas_EA, ntree = arg1, mtry = arg2)

  # Test du classifeur : classe predite
  rf_class <- predict(rf,datas_ET, type="response")

  # Matrice de confusion
  print(table(datas_ET$RESPONSE, rf_class))

  # Test du classifeur : probabilites pour chaque prediction
  rf_prob <- predict(rf, datas_ET, type="prob")

  # Courbe ROC
  rf_pred <- prediction(rf_prob[,2], datas_ET$RESPONSE)
  rf_perf <- performance(rf_pred,"tpr","fpr")
  plot(rf_perf, main = "Random Forests randomForest()", add = arg3, col = arg4)

  # Calcul de l'AUC et affichage par la fonction cat()
  rf_auc <- performance(rf_pred, "auc")
  cat("AUC = ", as.character(attr(rf_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()
  
  return (rf)
}


#---------------------#
# K-NEAREST NEIGHBORS #
#---------------------#

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC
test_knn <- function(arg1, arg2, arg3, arg4){
  # Apprentissage et test simultanes du classifeur de type k-nearest neighbors
  knn <- kknn(RESPONSE~., datas_EA, datas_ET, k = arg1, distance = arg2)
  
  # Matrice de confusion
  print(table(datas_ET$RESPONSE, knn$fitted.values))
  
  # Courbe ROC
  knn_pred <- prediction(knn$prob[,2], datas_ET$RESPONSE)
  knn_perf <- performance(knn_pred,"tpr","fpr")
  plot(knn_perf, main = "Classifeurs K-plus-proches-voisins kknn()", add = arg3, col = arg4)
  
  # Calcul de l'AUC et affichage par la fonction cat()
  knn_auc <- performance(knn_pred, "auc")
  cat("AUC = ", as.character(attr(knn_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()

  return (knn)
}


#-------------------------#
# SUPPORT VECTOR MACHINES #
#-------------------------#

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC
test_svm <- function(arg1, arg2, arg3){
  # Apprentissage du classifeur
  svm <- svm(RESPONSE~., datas_EA, probability=TRUE, kernel = arg1)

  # Test du classifeur : classe predite
  svm_class <- predict(svm, datas_ET, type="response")

  # Matrice de confusion
  print(table(datas_ET$RESPONSE, svm_class))

  # Test du classifeur : probabilites pour chaque prediction
  svm_prob <- predict(svm, datas_ET, probability=TRUE)

  # Recuperation des probabilites associees aux predictions
  svm_prob <- attr(svm_prob, "probabilities")

  # Courbe ROC 
  svm_pred <- prediction(svm_prob[,1], datas_ET$RESPONSE)
  svm_perf <- performance(svm_pred,"tpr","fpr")
  plot(svm_perf, main = "Support vector machines svm()", add = arg2, col = arg3)

  # Calcul de l'AUC et affichage par la fonction cat()
  svm_auc <- performance(svm_pred, "auc")
  cat("AUC = ", as.character(attr(svm_auc, "y.values")))
  
  # Return sans affichage sur la console
  invisible()

  return (svm)
}

#-----------------#
# NEURAL NETWORKS #
#-----------------#

# Definition de la fonction d'apprentissage, test et evaluation par courbe ROC
test_nnet <- function(arg1, arg2, arg3, arg4, arg5){
  # Redirection de l'affichage des messages intermédiaires vers un fichier texte
  sink('output.txt', append=T)
  
  # Apprentissage du classifeur 
  nn <- nnet(RESPONSE~., datas_EA, size = arg1, decay = arg2, maxit=arg3, MaxNWts=84581)
  
  # Réautoriser l'affichage des messages intermédiaires
  sink(file = NULL)
  
  # Test du classifeur : classe predite
  nn_class <- predict(nn, datas_ET, type="class")
  
  # Matrice de confusion
  print(table(datas_ET$RESPONSE, nn_class))

  # Test des classifeurs : probabilites pour chaque prediction
  nn_prob <- predict(nn, datas_ET, type="raw")
  
  # Courbe ROC 
  nn_pred <- prediction(nn_prob[,1], datas_ET$RESPONSE)
  nn_perf <- performance(nn_pred,"tpr","fpr")
  plot(nn_perf, main = "Réseaux de neurones nnet()", add = arg4, col = arg5)

  # Calcul de l'AUC
  nn_auc <- performance(nn_pred, "auc")
  cat("AUC = ", as.character(attr(nn_auc, "y.values")))

  # predict new datas
  
  
  # Return ans affichage sur la console
  invisible()

  return (nn)
}


#-------------------------------------------------#
# APPRENTISSAGE DES CONFIGURATIONS ALGORITHMIQUES #
#-------------------------------------------------#

# Arbres de decision
writeLines("\n\nArbres de décision: ")
test_rpart("gini", 10, FALSE, "red")
test_rpart("gini", 5, TRUE, "blue")
test_rpart("information", 10, TRUE, "green")
test_rpart("information", 5, TRUE, "orange")

# Forets d'arbres decisionnels aleatoires
writeLines("\n\nRandom Forest: ")
rf = test_rf(300, 3, FALSE, "red")
test_rf(300, 5, TRUE, "blue")
test_rf(500, 3, TRUE, "green")
test_rf(500, 5, TRUE, "orange")

# K plus proches voisins
writeLines("\n\nK-Nearest neighbors: ")
test_knn(10, 1, FALSE, "red")
test_knn(10, 2, TRUE, "blue")
test_knn(20, 1, TRUE, "green")
test_knn(20, 2, TRUE, "orange")

# Support vector machines
writeLines("\n\nSupport vector machines: ")
test_svm("linear", FALSE, "red")
test_svm("polynomial", TRUE, "blue")
test_svm("radial", TRUE, "green")
test_svm("sigmoid", TRUE, "orange")


# Réseaux de neurones nnet
writeLines("\n\nNeural Network: ")
test_nnet(10, 0.01, 100, FALSE, "red")
test_nnet(25, 0.01, 300, TRUE, "tomato")
test_nnet(25, 0.01, 100, TRUE, "blue")
test_nnet(25, 0.01, 300, TRUE, "purple")
test_nnet(50, 0.001, 100, TRUE, "green")
test_nnet(50, 0.001, 300, TRUE, "turquoise")

pred <- predict(rf, datas_new, type="class")
print(pred)