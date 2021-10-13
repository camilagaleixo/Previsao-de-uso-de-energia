# Projeto8 - Modelagem Preditiva em IoT - Previsão de uso de energia

# Definindo o diretório de trabalho
setwd("C:/Users/camil/OneDrive/Documentos/FCD/5.ML/Cap.20-Projetos_com_Feedback/Projeto8-R")
getwd()

###########################################

# ETAPA 1 - Definindo o problema de negócio

# Objetivo: Este projeto de IoT tem como objetivo a criação de modelo preditivo para a previsão 
# de consumo de energia de eletrodomésticos. Os dados utilizados incluem medições de sensores
# de temperatura e umidade de uma rede sem fio, previsão do tempo de uma estação de um aeroporto
# e uso de energia utilizada por luminárias.

# Variável target: Appliances

# Premissas: usar RandomForest para a seleção de atributos e SVM, Regressão 
# Logística Multilinear ou Gradient Boosting para o modelo preditivo. 

###########################################

# ETAPA 2 - Coleta dos dados
dados_treino <- read.csv("dados/projeto8-training.csv")
dados_teste <- read.csv("dados/projeto8-testing.csv")

###########################################

# ETAPA 3 - Análise Exploratória dos dados

View(dados_treino)
View(dados_teste)
str(dados_treino)

# Verificando a relação entre os dias da semana e o consumo de energia com eletrodomésticos

library(dplyr)
consumo_dia_semana <- 
  dados_treino%>%
  group_by(WeekStatus)%>%
  summarise(Consumo = mean(Appliances))

library(ggplot2)

# Gerando o gráfico 1:
ggplot(consumo_dia_semana, aes(y = Consumo, x = WeekStatus)) +
  geom_bar(stat = "identity", fill = "orange", width = .5) +
  ggtitle("Consumo x Dia da semana") +
  geom_text(aes(label = round(Consumo,2)), vjust = -.5) +
  scale_y_continuous(limits = c(0, 120))

# Gerando o gráfico 2 de outra forma:
dados_treino%>%
  group_by(Day_of_week)%>%
  summarise(Consumo = mean(Appliances))%>%
  ggplot(aes(y = Consumo, x = Day_of_week)) +
  geom_bar(stat = "identity", fill = "orange", width = .5) +
  labs(title = "Consumo x Dia da semana", x = "Dia da Semana", y = "Consumo") +
  geom_text(aes(label = round(Consumo,2)), vjust = -.5) +
  scale_y_continuous(limits = c(0, 120))

# Ajustando os níveis da coluna "Day_of_week" pra que o gráfico fique na ordem correta

levels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
dados_treino$Day_of_week <- factor(dados_treino$Day_of_week, levels = levels)

# Executando o gráfico novamente:
dados_treino%>%
  group_by(Day_of_week)%>%
  summarise(Consumo = mean(Appliances))%>%
  ggplot(aes(y = Consumo, x = Day_of_week)) +
  geom_bar(stat = "identity", fill = "orange", width = .5) +
  labs(title = "Consumo x Dia da semana", x = "Dia da Semana", y = "Consumo") +
  geom_text(aes(label = round(Consumo,2)), vjust = -.5) +
  scale_y_continuous(limits = c(0, 120))
  
# Verificando se há dados missing
sum(is.na(dados_treino))

# Verificando a correlação entre as variáveis
library(corrplot)

# Separando um novo dataframe somente com as variaveis do tipo numérico para verificar a correlação.
# A variável Day_of_week será tranformada em numérica também

dados_treino2 <- dados_treino[,-c(1, 31, 32)]
dados_treino2$Dia_da_semana <- as.numeric(factor(dados_treino$Day_of_week,
                                      label = 1:7,
                                      levels = levels))
View(dados_treino2)
str(dados_treino2)

corrplot(cor(dados_treino2), method = "square")

# No gráfico foi possível verificar que existe uma alta correlação  entre as variáveis "rv1" e "rv2",
# porém, o que parece é que ambas as colunas possuem a mesma informação. Dessa forma, será feita
# essa análise abaixo:
sum(dados_treino$rv1-dados_treino$rv2)
sum(dados_treino$rv1 == dados_treino$rv2)

# Verificando os dados estatísticos do dataset
summary(dados_treino)

# Analisando os dados estatísticos, parece existir dados outliers na variável target, 
# para fazer esse check, será verificado o histograma e dispersão destes dados

ggplot(dados_treino, aes(x = Appliances)) +
  geom_histogram(bins = 30, fill = "blue")+
  scale_x_continuous(limits = c(0,1100))

ggplot (dados_treino, aes(x = Appliances)) +
  geom_boxplot()+
  scale_x_continuous(limits = c(0, 1100))

# Pela leitura do gráfico é possível verificar que a maioria dos dados está concentrada
# até o valor 150. Dessa forma, será verificada a quantidade de dados acima de 150 e 300:
count(filter(dados_treino, Appliances > 150))
count(filter(dados_treino, Appliances > 300))

###########################################

# ETAPA 4 - Pré-processamento dos dados

# Como pode ser observado na análise exploratória, as variáveis rv1 e rv2 possuem os mesmos dados.
# Dessa forma, a variável rv2 será descartada. Além disso, foram verificados alguns outliers,
# mas os mesmos não serão removidos agora. Primeiro será analisado o modelo com os outliers e
# na fase de otimização, os mesmos serão desconsiderados.

# Para preparar o dataset para o treino, serão removidas as variáveis categóricas e mantidas apenas 
# as numéricas

# Removendo as variáveis dos datasets de treino e de teste
dados_train <- subset(dados_treino2, select = -rv2)
View(dados_train)
dados_test <- subset(dados_teste, select = -c(date, rv2, WeekStatus, Day_of_week))
dados_test$Dia_da_semana <- as.numeric(factor(dados_teste$Day_of_week,
                                                 label = 1:7,
                                                 levels = levels))
View(dados_test)

###########################################

# ETAPA 5 - Treinamento do modelo

# O treinamento será feito usando os algoritmos SVM, Regressão Logística Multilinear e
# Gradient Boosting. Será verificado qual é o melhor modelo.
# Porém, antes disso, será feita a seleção de atributos para verificar as melhores variáveis para 
# treinamento do modelo.

# Feature selection
library(randomForest)
library(caret)

modelo <- randomForest(Appliances ~ ., data = dados_train, importance = TRUE)
modelo

varImpPlot(modelo)

# Como pode ser visto no gráfico, a variável rv1 não possui muita importância para a medição
# da variável target, sendo assim, a mesma será retirada do dataset para realização do treinamento
dados_train2 <- subset(dados_train, select = -rv1)
dados_test2 <- subset(dados_test, select = -rv1)
View(dados_train2)
View(dados_test2)

# Treinamento do modelo usando o algoritmo SVM
library(e1071)

modelo_svm <- svm(Appliances ~ ., data = dados_train2, type = "eps-regression")

# Treinamento do modelo usando o algoritmo de Regressão Linear Múltipla
modelo_reg <- lm(Appliances ~., data = dados_train2)

# Treinamento do modelo usando o algoritmo Gradient Boosting
library(xgboost)

data = as.matrix(subset(dados_train2, select = -Appliances))
label = dados_train2$Appliances

modelo_boosting <- xgboost(data = data, label = label, nrounds = 2)

###########################################

# ETAPA 6 - Avaliação do modelo

# A avaliação do modelo será feita usando a métrica RMSE (Root Mean Squared Error)

# SVM
previsao_svm <- predict(modelo_svm, dados_test2[-1])

RMSE(dados_test2$Appliances, previsao_svm)

# Regressão Linear Múltipla
previsao_reg <- predict(modelo_reg, dados_test2[-1])

RMSE(dados_test2$Appliances, previsao_reg)

# Gradient Boosting
newdata <- as.matrix(dados_test2[-1])

previsao_boosting <- predict(modelo_boosting, newdata = newdata)

RMSE(dados_test2$Appliances, previsao_boosting)

# Dos 3 modelos, o que apresentou menor erro foi o SVM.

###########################################

# ETAPA 7 - Otimização do modelo

# Nesta etapa de otimização serão feitas duas modificações:
# 1) Normalização dos dados
# 2) Remoção dos valores outliers

# 1) Normalizando os dados
dados_train3 <- data.frame(scale(dados_train2))
View(dados_train3)
dados_test3 <- data.frame(scale(dados_test2))
View(dados_test3)

# Calculando novamente os modelos
# SVM
modelo_svm2 <- svm(Appliances ~ ., data = dados_train3, type = "eps-regression")
previsao_svm2 <- predict(modelo_svm2, dados_test3[-1])
RMSE(dados_test3$Appliances, previsao_svm2)

# Regressão Linear Múltipla
modelo_reg2 <- lm(Appliances ~., data = dados_train3)
previsao_reg2 <- predict(modelo_reg2, dados_test3[-1])
RMSE(dados_test3$Appliances, previsao_reg2)

# Gradient Boosting
data = as.matrix(subset(dados_train3, select = -Appliances))
label = dados_train3$Appliances
modelo_boosting2 <- xgboost(data = data, label = label, nrounds = 2)

newdata <- as.matrix(dados_test3[-1])
previsao_boosting2 <- predict(modelo_boosting2, newdata = newdata)
RMSE(dados_test3$Appliances, previsao_boosting2)

# Com os dados normalizados o RMSE foi muito menor para os 3 algoritmos.
# No entanto, o que ainda está apresentando melhor desempenho é o SVM.
# Seguindo com a etapa de otimização, os outliers observados na etapa de análise serão removidos.

# 2) Remoção dos outliers
dados_train4 <- filter(dados_train2, Appliances<150)
View(dados_train4)

ggplot (dados_train4, aes(x = Appliances)) +
  geom_boxplot()+
  scale_x_continuous(limits = c(0, 1100))

# Aplicando novamente a normalização sobre o novo conjunto de dados
dados_train5 <- data.frame(scale(dados_train4))

# Calculando novamente os modelos
# SVM
modelo_svm3 <- svm(Appliances ~ ., data = dados_train5, type = "eps-regression")
previsao_svm3 <- predict(modelo_svm3, dados_test3[-1])
RMSE(dados_test3$Appliances, previsao_svm3)

# Regressão Linear Múltipla
modelo_reg3 <- lm(Appliances ~., data = dados_train5)
previsao_reg3 <- predict(modelo_reg3, dados_test3[-1])
RMSE(dados_test3$Appliances, previsao_reg3)

# Gradient Boosting
data = as.matrix(subset(dados_train5, select = -Appliances))
label = dados_train5$Appliances
modelo_boosting3 <- xgboost(data = data, label = label, nrounds = 2)

newdata <- as.matrix(dados_test3[-1])
previsao_boosting3 <- predict(modelo_boosting3, newdata = newdata)
RMSE(dados_test3$Appliances, previsao_boosting3)

# Nesta última otimização foi possível verificar que o desempenho dos modelos foi menor,
# Logo, o melhor modelo para este projeto é o modelo_svm2, que apresentou um RMSE de 0,87.