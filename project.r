datas <- read.csv("Data Projet.csv", header = TRUE, sep = ",", dec = ".", stringsAsFactors = T) 
datas_new <- read.csv("Data Projet New.csv", header = TRUE, sep = ",", dec = ".", stringsAsFactors = T) 
library(arules)
library(arulesViz)


# Generation des regles d'association contenant au moins 2 items pour minsupp=15% et minconf=50%
rules2 <- apriori(datas, parameter = list(supp = 0.15, conf = 0.5, target ="rules", minlen=2))


#plot(rules2, method="graph", engine='interactive', shading=NA)

#rules2 <- sort(rules2, by="confidence", decreasing=TRUE)

df_rules2 <- data.frame(inspect(rules2))
View(df_rules2)
summary(df_rules2)
str(df_rules2)
names(df_rules2)

# Cluster
# Installation/mise à jour des librairies
install.packages("cluster")
install.packages("ggplot2")
install.packages("CORElearn")

# Activation des librairies
library(cluster)
library(ggplot2)
library(CORElearn)

# Calcul de la matrice de distance par la fonction daisy() pour variables hétérogènes
dmatrix <- daisy(datas)

# Résumé de la matrice
summary(dmatrix)

km4 <- kmeans(dmatrix, 4)
table(km4$cluster, datas$RESPONSE)

qplot(km4$cluster, data=datas, fill=RESPONSE)
qplot(ED, as.factor(km4$cluster), data=datas, color=RESPONSE) + geom_jitter(width = 0.3, height = 0.3)
qplot(CAR, as.factor(km4$cluster), data=datas, color=RESPONSE) + geom_jitter(width = 0.3, height = 0.3)
qplot(INCCAT, as.factor(km4$cluster), data=datas, color=RESPONSE) + geom_jitter(width = 0.3, height = 0.3)
qplot(INCOME, as.factor(km4$cluster), data=datas, color=RESPONSE) + geom_jitter(width = 0.3, height = 0.3)

datas_km4 <- data.frame(datas, km4$cluster)


# SELECTION DATTRIBUTS

# Calcul de l'Entropie de Shannon
entropy <- function(variable) {
  # Fréquences des valeurs 
  frequences <- table(variable)/length(variable)
  # Vectorisation des valeurs
  vecteur <- as.data.frame(frequences)[,2]
  # Suppression des 0 pour éviter les NaN résultant du log2()
  vecteur <- vecteur[vecteur>0]
  # Calcul de la valeur de l'entropie
  return(-sum(vecteur * log2(vecteur)))
}


entropy(datas$RESPONSE)

datas$RESPONSE

# Calcul des Information Gain pour la prédiction de Produit
attrEval(RESPONSE~., datas, estimator = "InfGain")

# Liste des noms de variables
ListeVar = attr(attrEval(RESPONSE~., datas, estimator = "InfGain"), "names")

# Liste des valeurs d'Information Gain
ListeVal = as.vector(attrEval(RESPONSE~., datas, estimator = "InfGain"))

# Affichage par ordre croissant d'IG
data.frame(ListeVar, ListeVal)[order(ListeVal),]

# Fonction d'affichage en table triée par valeurs décroissantes
affEval <- function(mesure) {
  result <- attrEval(RESPONSE~., datas, estimator = mesure)
  ListeVar = attr(result, "names")
  ListeVal = as.vector(result)
  data.frame(ListeVar, ListeVal)[order(ListeVal, decreasing = T),]
}

# Gain Ratio
affEval("InfGain")
affEval("Gini")
affEval("Relief")
affEval("MDL")
affEval("GainRatio")

# !!!!!!!!!!!
# TODO LA SUITE AVEC LA DISCRETISATION NE FONCTIONNE PAS
# !!!!!!!!!!!

library(arules)

# Discrétisations en largeur
datas$INCOME_DW3 = discretize(datas$INCOME, method="interval", breaks = 3)
datas$INCOME_DW5 = discretize(datas$INCOME, method="interval", breaks = 5)
datas$INCCAT_DW3 = discretize(datas$INCCAT, method="interval", breaks = 3)
datas$INCCAT_DW5 = discretize(datas$INCCAT, method="interval", breaks = 5)

# Calcul des mesures
affEval("InfGain")
affEval("GainRatio")
affEval("Gini")
affEval("Relief")
affEval("MDL")

# Discrétisations en profondeur
datas$INCOME_DF3 = discretize(datas$INCOME, method="frequency", breaks = 3)
datas$INCOME_DF5 = discretize(datas$INCOME, method="frequency", breaks = 5)
datas$INCCAT_DF3 = discretize(datas$INCCAT, method="frequency", breaks = 3)
datas$INCCAT_DF5 = discretize(datas$INCCAT, method="frequency", breaks = 5)

# Calcul des mesures
affEval("InfGain")
affEval("GainRatio")
affEval("Gini")
affEval("Relief")
affEval("MDL")

#--------------------------#
# Histogrammes d'Effectifs #
#--------------------------#

# Affichage des histogrammes d'effectifs avec classe en couleur
library(ggplot2)

# Fonction d'affichage d'un histogramme d'effectif avec classe en couleur
# nom_var : nom de la variable à afficher sous la forme (ex: nom_data_frame$nom_variable)
# lib_var : nom de la variable sous forme de chaîne de caractères (ex: "nom_variable")
affHist <- function(nom_var, lib_var) {
  qplot(nom_var, data=datas, fill=RESPONSE, 
        main=paste("Distribution de", lib_var),      # Titre avec nom de la variable
        xlab=paste("Valeur de ", lib_var),           # Axe des absisses avec nom de la variable
        ylab="Nombre d'exemples",  
  )
}

# Histogrammes des discrétisations de INCOME
affHist(datas$INCOME_DW3, "INCOME_DW3")
affHist(datas$INCOME_DW5, "INCOME_DW5")
affHist(datas$INCOME_DF3, "INCOME_DF3")
affHist(datas$INCOME_DF5, "INCOME_DF5")

# Histogrammes des discrétisations de INCCAT
affHist(datas$INCCAT_DW3, "INCCAT_DW3")
affHist(datas$INCCAT_DW5, "INCCAT_DW5")
affHist(datas$INCCAT_DF3, "INCCAT_DF3")
affHist(datas$INCCAT_DF5, "INCCAT_DF5")
