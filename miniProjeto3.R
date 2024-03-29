## Mini projeto 3

# Versão do R utilizada no projeto
R.version

# Diretório do projeto
setwd("C:/Users/wgnr2/Desktop/Curso PowerBI DSA/cap15")
getwd()

# Pacotes
install.packages("Amelia")
install.packages("caret")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("reshape")
install.packages("randomForest")
install.packages("e1071")

# Carregando os pacotes
library("Amelia")
library("caret")
library("ggplot2")
library("dplyr")
library("reshape")
library("randomForest")
library("e1071")

# dataset : https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients#
dados_clientes <- read.csv("dados/dataset.csv")

# visualização dos dados
View(dados_clientes)
dim(dados_clientes)
str(dados_clientes)
summary(dados_clientes)

### Análise explorátoria, limpeza e transformação ###

# removendo a coluna ID
dados_clientes$ID <- NULL
dim(dados_clientes)
View(dados_clientes)

# renomear a coluna de classe
colnames(dados_clientes)
colnames(dados_clientes)[24] <- "Inadimplente"
colnames(dados_clientes)
View(dados_clientes)

# Verificando valores ausentes e removendo do dataset
sapply(dados_clientes, function(x) sum(is.na(x))) #similar a LAMBDA
?missmap
missmap(dados_clientes, main = "Valores Missing Observados")
dados_clientes <- na.omit(dados_clientes)

# convertendo os atributos genero, escolaridade, estado civil e idade
# para fatores (categorias)

# renomeando as colunas categóricas
colnames(dados_clientes)
colnames(dados_clientes)[2] <- "Genero"
colnames(dados_clientes)[3] <- "Escolaridade"
colnames(dados_clientes)[4] <- "Estado_civil"
colnames(dados_clientes)[5] <- "Idade"
colnames(dados_clientes)

# Genero
View(dados_clientes$Genero)
str(dados_clientes$Genero)
summary(dados_clientes$Genero)
?cut
dados_clientes$Genero <- cut(dados_clientes$Genero,
                             c(0,1,2),
                             labels = c("Masculino",
                                        "Feminino"))

View(dados_clientes$Genero)
str(dados_clientes$Genero)
summary(dados_clientes$Genero)

# Escolaridade
str(dados_clientes$Escolaridade)
summary(dados_clientes$Escolaridade)
dados_clientes$Escolaridade <- cut(dados_clientes$Escolaridade,
                      c(0,1,2,3,4),
                      labels = c("Pos Graduado",
                                 "Graduado",
                                 "Ensino Medio",
                                 "Outros"))

View(dados_clientes$Escolaridade)
str(dados_clientes$Escolaridade)
summary(dados_clientes$Escolaridade)

# Estado Civil
str(dados_clientes$Estado_civil)
summary(dados_clientes$Estado_civil)
dados_clientes$Estado_civil <- cut(dados_clientes$Estado_civil,
                                   c(-1,0,1,2,3),
                                   labels = c("Desconhecido",
                                              "Casado",
                                              "Solteiro",
                                              "Outro"))

View(dados_clientes$Estado_civil)
str(dados_clientes$Estado_civil)
summary(dados_clientes$Estado_civil)

# Convertendo a variável para o tipo fator com faixa etária
str(dados_clientes$Idade)
summary(dados_clientes$Idade)
hist(dados_clientes$Idade)
dados_clientes$Idade <- cut(dados_clientes$Idade,
                      c(0,30,50,100),
                      labels = c("Jovem",
                                 "Adulto",
                                 "Idoso"))

View(dados_clientes$Idade)
str(dados_clientes$Idade)
summary(dados_clientes$Idade)

# Convertendo as variáveis de pagamento para o tipo fator
dados_clientes$PAY_0 <- as.factor(dados_clientes$PAY_0)
dados_clientes$PAY_2 <- as.factor(dados_clientes$PAY_2)
dados_clientes$PAY_3 <- as.factor(dados_clientes$PAY_3)
dados_clientes$PAY_4 <- as.factor(dados_clientes$PAY_4)
dados_clientes$PAY_5 <- as.factor(dados_clientes$PAY_5)
dados_clientes$PAY_6 <- as.factor(dados_clientes$PAY_6)

# Dataset após as conversões
str(dados_clientes)
sapply(dados_clientes, function(x) sum(is.na(x)))
missmap(dados_clientes, main = "Valores Missing Observados")
dados_clientes <- na.omit(dados_clientes)
missmap(dados_clientes, main = "Valores Missing Observados")
dim(dados_clientes)

# Alterando a variável dependente(Target) para o tipo fator
str(dados_clientes$Inadimplente)
colnames(dados_clientes)
dados_clientes$Inadimplente <- as.factor(dados_clientes$Inadimplente)
str(dados_clientes$Inadimplente)
View(dados_clientes)

# Total de inadimplentes versus não-inadimplentes
table(dados_clientes$Inadimplente)

# proporção
prop.table(table(dados_clientes$Inadimplente))

# Plot da distribuição 
qplot(Inadimplente, data = dados_clientes, geom = "bar") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


# Set seed
set.seed(12345)

# amostragem estratificada
# seleciona as linhas de acordo com a variável inadimplente como strata
# técnica de Hold-out
install.packages('Rcpp')
library('Rcpp')
??createDataPartition
indice <- createDataPartition(dados_clientes$Inadimplente, p = 0.75, list = FALSE)
dim(indice)

# definimos os dados de treinamento como subconjunto do conjunto de dados original
# com números de índice de linha (conforme identificado acima) e todas as colunas
dados_treino <- dados_clientes[indice,]
colnames(dados_treino)
dim(dados_treino)
table(dados_treino$Inadimplente)

# proporção entre as classes
prop.table(table(dados_treino$Inadimplente))

# Número de registros no dataset de treinamento
dim(dados_treino)

# Comparamos as porcentagens entre as classes de treinamento e dados originais
compara_dados <- cbind(prop.table(table(dados_treino$Inadimplente)),
                       prop.table(table(dados_clientes$Inadimplente)))
colnames(compara_dados) <- c("Treinamento", "Original")
compara_dados

# Melt Data - Converte colunas em linhas
?reshape2::melt
melt_compara_dados <- melt(compara_dados)
melt_compara_dados

# Plot para ver a distribuição do treinamento vs Original
ggplot(melt_compara_dados, aes(x = X1, y = value)) +
  geom_bar(aes(fill = X2), stat = "identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Tudo o que não está no dataset de treino esta no dataset de teste (-)
dados_teste <- dados_clientes[-indice,]
dim(dados_teste)
dim(dados_treino)


############### Modelo de Machine Learning #################

# Primeiro versão (sem balanceamento dos dados)
?randomForest
View(dados_treino)
modelo_v1 <- randomForest(Inadimplente ~ .,data = dados_treino)
modelo_v1

# Avaliando o modelo
plot(modelo_v1)

# Previsões com dados de teste
previsoes_v1 <- predict(modelo_v1, dados_teste)

# Matriz de confusão
?caret::confusionMatrix #documentação
cm_v1 <- caret::confusionMatrix(previsoes_v1, dados_teste$Inadimplente,
                                positive = "1")
View(cm_v1)

# Calculando Precision, Recall e F1-Score do modelo preditivo
y <- dados_teste$Inadimplente
y_pred_v1 <- previsoes_v1

precision <- posPredValue(y_pred_v1, y)
precision

recall <- sensitivity(y_pred_v1, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

# Balancemento de classe com SMOTE - Synthetic Minority Over-sampling Technique
# https://arxiv.org/pdf/1106.1813.pdf

# carregamento das dependencias + pacote
install.packages(c("zoo","xts","quantmod", 'ROCR')) # Dependencias do DMwr
install.packages( "C:/Users/wgnr2/Downloads/DMwR_0.4.1.tar.gz", repos=NULL, type="source" )
#install.packages("DMwR") # versão depreciada, removida do repo oficial do R
library("DMwr")
?SMOTE

# proporção antes do balanceamento
table(dados_treino$Inadimplente)
prop.table(table(dados_treino$Inadimplente))

# Aplicando o SMOTE
set.seed(9560)
dados_treino_bal <- SMOTE(Inadimplente ~ ., data = dados_treino)

# Proporção depois do balanceamento
table(dados_treino_bal$Inadimplente)
prop.table(table(dados_treino_bal$Inadimplente))


# Segunda versão do modelo (dados balanceados)
modelo_v2 <- randomForest(Inadimplente ~ ., data = dados_treino_bal)
modelo_v2

# Avaliando o modelo
plot(modelo_v2)

# Previsões com dados de teste
previsoes_v2 <- predict(modelo_v2, dados_teste)

# Matriz de confusão
?caret::confusionMatrix #documentação
cm_v2 <- caret::confusionMatrix(previsoes_v2, dados_teste$Inadimplente,
                                positive = "1")
View(cm_v2)

# Calculando Precision, Recall e F1-Score do modelo preditivo
y <- dados_teste$Inadimplente
y_pred_v2 <- previsoes_v2

precision <- posPredValue(y_pred_v2, y)
precision

recall <- sensitivity(y_pred_v2, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

# Importância das variáveis preditoras
View(dados_treino_bal)
varImpPlot(modelo_v2)

# Obtendo as variáveis mais importantes
imp_var <- importance(modelo_v2)
varImportance <- data.frame(Variables = row.names(imp_var),
                            importance = round(imp_var[ ,'MeanDecreaseGini'],2))
                            
# Criando o rank de variáveis baseado na importância
rankImportance <- varImportance %>%
  mutate(rank = paste0('#', dense_rank(desc(importance))))

# plotando...
ggplot(rankImportance,
       aes(x = reorder(Variables, importance),
           y = importance,
           fill = importance)) +
  geom_bar(stat='identity') +
  geom_text(aes(x= Variables, y = 0.5, label = rank),
            hjust = 0,
            vjust = 0.55,
            size = 4,
            colour = 'red') +
  labs(x = 'Variables') +
coord_flip()
       
### Versão final do modelo ###

# construção baseada no Ranking

colnames(dados_treino_bal)
modelo_v3 <- randomForest(Inadimplente ~ PAY_0 + PAY_2 + PAY_3
                          + PAY_AMT1 + PAY_AMT2 + PAY_5
                          + BILL_AMT1,
                          data = dados_treino_bal)
modelo_v3


# Avaliando o modelo
plot(modelo_v3)

# Previsões com dados de teste
previsoes_v3 <- predict(modelo_v3, dados_teste)

# Matriz de confusão
?caret::confusionMatrix #documentação
cm_v3 <- caret::confusionMatrix(previsoes_v3, dados_teste$Inadimplente,
                                positive = "1")
View(cm_v3)

# Calculando Precision, Recall e F1-Score do modelo preditivo
y <- dados_teste$Inadimplente
y_pred_v3 <- previsoes_v3

precision <- posPredValue(y_pred_v3, y)
precision

recall <- sensitivity(y_pred_v3, y)
recall

F1 <- (2 * precision * recall) / (precision + recall)
F1

# Salvando o modelo treinado em disco
saveRDS(modelo_v3, file = "modelo/modelo_v3.rds")

# Carregando o modelo
modelo_final <- readRDS("modelo/modelo_v3.rds")


# previsões com novos dados

# dados do clientes
PAY_0 <- c(0,0,0)
PAY_2 <- c(0,0,0)
PAY_3 <- c(1,0,0)
PAY_AMT1 <- c(1100, 1000, 1200)
PAY_AMT2 <- c(1500, 1300, 1150)
PAY_5 <- c(0, 0, 0)
BILL_AMT1 <- c(350, 420, 280)

# Concatena em um dataframe
novos_clientes <- data.frame(PAY_0, PAY_2, PAY_3, PAY_AMT1, PAY_AMT2,
                             PAY_5, BILL_AMT1)
View(novos_clientes)

# Previsões (resulta em erro pois os tipos de dados dos dados de entrada
# não são iguais aos utilizados no treinamento do modelo)
previsoes_novos_clientes <- predict(modelo_final, novos_clientes)

# checando os tipos de dados usados no modelo
str(dados_treino_bal) # fator
str(novos_clientes) # numericos

# convertendo os tipos de dados
novos_clientes$PAY_0 <- factor(novos_clientes$PAY_0, levels(dados_treino_bal$PAY_0))
novos_clientes$PAY_2 <- factor(novos_clientes$PAY_2, levels(dados_treino_bal$PAY_2))
novos_clientes$PAY_3 <- factor(novos_clientes$PAY_3, levels(dados_treino_bal$PAY_3))
novos_clientes$PAY_5 <- factor(novos_clientes$PAY_5, levels(dados_treino_bal$PAY_5))
str(novos_clientes)

# Previsões
previsoes_novos_clientes <- predict(modelo_final, novos_clientes)
previsoes_novos_clientes <- data.frame()
View(previsoes_novos_clientes)

novos_clientes_predict <- data.frame(PAY_0, PAY_2, PAY_3, PAY_AMT1, PAY_AMT2,
                             PAY_5, BILL_AMT1,previsoes_novos_clientes)
colnames(novos_clientes_predict)
colnames(novos_clientes_predict)[8] <- "Inadimplente"

View(novos_clientes_predict)















