#Manipulação dos dados
import pandas as pd

#Gráficos
import seaborn as sns
import matplotlib.pyplot as plt

#Processa outliers
from scipy.stats.mstats import winsorize

#Pré-Processamento
from sklearn.preprocessing import StandardScaler

#Calculos matemáticos
import numpy as np

#Avaliação do modelo
from sklearn.model_selection import RandomizedSearchCV,StratifiedKFold,cross_val_score

#Auxilia na avaliação do modelo
from scipy.stats import uniform, randint

#Separação dos dados em treino e teste
from sklearn.model_selection import train_test_split

#Mátricas de avaliação
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, classification_report

#Calculo da curva ROC
from sklearn.metrics import roc_curve, auc, confusion_matrix

#Salvar o Modelo
import joblib
import os

plt.style.use("ggplot")
class TratamentoDados:
    def __init__(self):
        pass

    @staticmethod
    def tratamentoValoresAnosTrabalhoAtual(anos):
        """
        Trata valores da coluna 'Years in current job' de forma simples:
        - NaN → None
        - Contém "10" → 10
        - Começa com número → pega o primeiro dígito
        - Começa com "<" → 0
        """
        if(pd.isna(anos)):
            return None
        elif("10" in anos):
            return 10
        elif(anos[0].isdigit()):
            return int(anos[0])
        elif(anos[0] == "<"):
            return 0
    

    def aplicarWinsorizacao(self, df, colunas=None, limites=(0, 0.01)):
        """
        Aplica winsorização nas colunas especificadas do DataFrame.
        
        Parâmetros:
        -----------
        df : pd.DataFrame
            DataFrame contendo os dados
        colunas : list, opcional
            Lista de colunas para aplicar winsorização. Se None, usa colunas padrão.
        limites : tuple, opcional
            Tupla com os limites inferior e superior para winsorização (padrão: 5% em cada extremidade)
            
        Retorna:
        --------
        pd.DataFrame
            DataFrame com as colunas winsorizadas
        """
    
        
        # Cria uma cópia para não modificar o original
        dfTratado = df.copy()
        
        # Aplica winsorização para cada coluna
        for coluna in colunas:
            if coluna in dfTratado.columns:
                dfTratado[coluna] = winsorize(dfTratado[coluna], limits=limites)
            
                
        return dfTratado
    
    

class AvalicaoModelo:
    def __init__(self):
        pass

    def predicaoModelosArvores(self, modelo, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


        model = modelo(random_state=42, class_weight = "balanced", max_depth=7)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acuracia = accuracy_score(y_test, y_pred)
        rc = recall_score(y_test, y_pred)
        precision =precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print("Modelo ", modelo)
        print(f"Acurácia: {acuracia:.4f}")
        print(f"Recall: {rc:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")



    def predicaoModeloRobustos(self, modelo, X, y):
        # Dividir os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        contagemClasses = y_train.value_counts()
        negativos = contagemClasses[0]
        positivos = contagemClasses[1]
        scale_pos_weight = negativos/positivos #servi para o modelo dar um foco maior na classe com menor numeros de dados na target
        modeloNome = modelo.__name__ #Pega o nome do modelo
    
        if modeloNome == "LGBMClassifier":
            model = modelo(scale_pos_weight=scale_pos_weight,learning_rate=0.1,random_state=42,verbose=-1)  # Silencia os logs do LightGBM
        else:
            model = modelo(scale_pos_weight=scale_pos_weight, learning_rate=0.1, random_state=42)
        
        #Treina com os dados
        model.fit(X_train, y_train)

        # Fazer previsões e calcular métricas
        y_pred = model.predict(X_test)

        #Calcula metricas principais
        acuracia = accuracy_score(y_test, y_pred)
        rc = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision =precision_score(y_test, y_pred)
        
        #Mostra os resultados
        print("Modelo ", modelo)
        print(f"Acurácia: {acuracia:.4f}")
        print(f"Recall: {rc:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(classification_report(y_test, y_pred))

    
    def avaliacaoFinalDoModelo(self, melhorEstimador, X_test, y_test):
        y_pred = melhorEstimador.predict(X_test)

        acuracia = accuracy_score(y_test, y_pred)
        rc = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision =precision_score(y_test, y_pred)


        print(f"Acurácia: {acuracia:.4f}")
        print(f"Recall: {rc:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(classification_report(y_test, y_pred))



    def predicaoModelosLogistico(self, modelo, X, y):
        # Dividir os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Aplicar logaritmo nos dados (adicionando 1 para evitar log(0))
        X_train_log = np.log1p(X_train)  # log1p = log(x + 1)
        X_test_log = np.log1p(X_test)
        
        # Padronizar os dados
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_log)
        X_test_scaled = scaler.transform(X_test_log)
        
        # Criar e treinar o modelo
        model = modelo(class_weight='balanced', max_iter=1000, solver='lbfgs')
        model.fit(X_train_scaled, y_train)
        
        # Fazer previsões e calcular métricas
        y_pred = model.predict(X_test_scaled)
        
        acuracia = accuracy_score(y_test, y_pred)
        rc = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        
        print("Modelo ", modelo)
        print(f"Acurácia: {acuracia:.4f}")
        print(f"Recall: {rc:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(classification_report(y_test, y_pred))

    

class Graficos:
    def __init__(self):
        pass  

    def relacaoDeVariaveisComTargetBoxplot(self, df, colunas, ncols, nrows):

        fig, axs = plt.subplots(figsize = (18, 8), ncols=ncols, nrows=nrows)
        if nrows * ncols == 1:
            axs = [axs]
        for i, coluna in enumerate(colunas):
            sns.boxplot(data=df, x="Loan Status", y = coluna, ax=axs[i])


        plt.suptitle("Diferença entre variavies em relação a target")
        plt.tight_layout()
        plt.show()

    def analiseUnivariada(self, df, tipo):
        colunas = df.select_dtypes(include=tipo).columns
        if tipo == "number": #Grafico especificos para colunas numericas
            fig, ax = plt.subplots(figsize=(15, 8), ncols=4, nrows=3)
            ax=ax.flatten()
            plt.suptitle("Distribuição das colunas numéricas")
            for i, c in enumerate(colunas):
                ax[i].hist(df[c])
                ax[i].set_xlabel(c)
            plt.tight_layout()
            plt.show()
        else:
            fig, ax = plt.subplots(figsize=(15, 10), ncols=2, nrows=2)
            ax=ax.flatten()
            plt.suptitle("Distribuição das colunas numéricas")
            for i, c in enumerate(colunas):
                quant_coluna = df[c].value_counts().reset_index()
                quant_coluna.columns = [c, "Quantidade"]
                sns.barplot(data=quant_coluna, x = c, y="Quantidade", ax=ax[i])
                ax[i].set_xlabel(c)
                ax[i].set_xticklabels(quant_coluna[c], rotation=45, ha='right')
                for j, valor in enumerate(quant_coluna["Quantidade"]):
                    ax[i].text(j, valor, str(valor), ha = "center", va = "bottom", fontsize = 10)#Adiciona os valores nos topos das barras
            plt.tight_layout()
            plt.show()

    def analiseBivariada(self, df, tipo):
        colunas = df.select_dtypes(include=tipo).columns
        if tipo == "number": #Grafico especificos para colunas numericas
            fig, axs = plt.subplots(figsize = (15, 8), ncols=4, nrows=3)
            axs = axs.flatten()
            plt.suptitle("Relação da média das colunas numericas com o status do Empréstimos")
            for i, col in enumerate(colunas):
                relacao_cols_target = df[["Loan Status", col]].groupby("Loan Status").mean().reset_index()
                relacao_cols_target.columns = ["Loan Status", "Media"]
                sns.barplot(x=relacao_cols_target["Loan Status"], y=relacao_cols_target["Media"], ax=axs[i])
                axs[i].set_xlabel("Loan Status")
                axs[i].set_ylabel(f"Media: {col}")

            plt.tight_layout()
            plt.show()
        else:
            fig, axs = plt.subplots(figsize = (15, 6), ncols=3, nrows=1)
            axs = axs.flatten()
            plt.suptitle("Relação das colunas categóricas com o status do Empréstimos")
            for i, col in enumerate(colunas[1:]):
                sns.countplot(x=col, hue='Loan Status', data=df, ax=axs[i])
                axs[i].set_xlabel("Loan Status")
                axs[i].set_ylabel(col)
                axs[i].tick_params(axis='x', rotation=90)

            plt.tight_layout()
            plt.show()
    
    def analisePercentualCategoricasTarget(self, df):

        # Função para adicionar rótulos de porcentagem
        def add_percentage_labels(ax, precision=1):
            for container in ax.containers:
                labels = [f'{w:.{precision}f}%' if w > 4 else '' for w in container.datavalues]
                ax.bar_label(container, labels=labels, label_type='center', 
                            fontsize=8, color='white', fontweight='bold')

        # Seleciona apenas colunas categóricas (excluindo a target se necessário)
        colunas_categoricas = df.select_dtypes(include='object').columns
        # Remove a coluna target se estiver incluída
        colunas_categoricas = [col for col in colunas_categoricas if col != 'Loan Status']

        fig, axs = plt.subplots(nrows=1, ncols=3, 
                            figsize=(15, 5))
        axs = axs.flatten()

        plt.suptitle("Proporção de Inadimplência por Variáveis Categóricas")

        for i, col in enumerate(colunas_categoricas):
            if i < len(axs):
                # Cria gráfico de barras empilhadas com porcentagem
                sns.histplot(
                    data=df,
                    x=col,
                    hue="Loan Status",
                    stat="percent",
                    multiple="fill",
                    shrink=0.8,
                    edgecolor='white',
                    linewidth=1,
                    ax=axs[i]
                )
            
                axs[i].set_title(f'{col}', fontweight='bold', fontsize=12)
                axs[i].set_xlabel("")
                axs[i].set_ylabel("Proporção (%)", fontsize=10)
                axs[i].tick_params(axis='x', rotation=90)

                # Adiciona rótulos de porcentagem
                add_percentage_labels(axs[i])
        plt.tight_layout()
        plt.show()

    def analiseUnivariadaBoxPlot(self,df, colunas):
        fig, ax = plt.subplots(figsize=(15, 5), ncols=3, nrows=1)
        ax=ax.flatten()
        plt.suptitle("Colunas numericas outliers")
        for i, c in enumerate(colunas):
            sns.boxenplot(df[c], ax=ax[i])
            ax[i].set_xlabel(c)
        plt.tight_layout()
        plt.show()

    
    def matrizCorrelacao(self, dfAnaliseExploratoria):
        """
        Vai mostra a correlação das variaveis numéricas
        """

        corr = dfAnaliseExploratoria.select_dtypes(include="number").corr()
        plt.figure(figsize=(15, 5))
        sns.heatmap(corr, vmin=-1, vmax=1, annot=True)
        plt.show()
    
    def curvaRoc(self, estimadorFinal, X_test, y_test):
        #Prever probabilidades para a curva ROC
        y_pred_prob = estimadorFinal.predict_proba(X_test)[:, 1]  # Probabilidades da classe positiva

        #Calcular a curva ROC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        #Plotar a curva ROC
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
        plt.xlabel('Taxa de Falsos Positivos (FPR)')
        plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
        plt.title('Curva ROC')
        plt.legend(loc='lower right')
        plt.show()

    def matrizConfusao(self, estimadorFinal, X_test, y_test):
        #Calcular a matriz de confusão
        y_pred = estimadorFinal.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        # 8. Plotar a matriz de confusão
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Classe 0', 'Classe 1'],
                    yticklabels=['Classe 0', 'Classe 1'])
        plt.xlabel('Predito')
        plt.ylabel('Verdadeiro')
        plt.title('Matriz de Confusão')
        plt.show()

    def importanciaVariaveis(self, X_train, estimadorFinal):
        colunasQueInfluenciamNaDecisão = {
        "Colunas" : X_train.columns,
        "Influencia" : estimadorFinal.feature_importances_
        }
        colunasQueInfluenciamNaDecisão = pd.DataFrame(colunasQueInfluenciamNaDecisão)
        ordenadoPorColunasQueInfluenciamNaDecisão = colunasQueInfluenciamNaDecisão.sort_values(by = "Influencia", ascending=False)


        influencias = ordenadoPorColunasQueInfluenciamNaDecisão["Influencia"] / ordenadoPorColunasQueInfluenciamNaDecisão["Influencia"].max()

        # Criar um gradiente de cores (quanto maior a importância, mais escura a barra)
        colors = plt.cm.Greens(influencias)  # Usando o colormap 'Blues'

        # Plotar o gráfico de barras com cores baseadas na importância
        plt.figure(figsize=(10, 6))
        bars = plt.bar(ordenadoPorColunasQueInfluenciamNaDecisão["Colunas"], ordenadoPorColunasQueInfluenciamNaDecisão["Influencia"] , color=colors)
        plt.xticks(range(len(ordenadoPorColunasQueInfluenciamNaDecisão["Colunas"])), ordenadoPorColunasQueInfluenciamNaDecisão["Colunas"].values, rotation=90)
        plt.xlabel('Variáveis')
        plt.ylabel('Importância')
        plt.title('Importância das Variáveis no Modelo (Cores por Influência)')

class Modelo():
    def __init__(self):
        pass

    def salvarModelo(self, modelo, caminhoArquivo):
        """
        Salva o modelo em um arquivo especificado.

        Args:
            modelo: O modelo de machine learning a ser salvo.
            caminhoArquivo (str): Caminho completo para salvar o modelo.
        """
        diretorio = os.path.dirname(caminhoArquivo)
        if diretorio and not os.path.exists(diretorio):
            os.makedirs(diretorio)
        joblib.dump(modelo, caminhoArquivo)
        print(f"Modelo salvo em: {caminhoArquivo}")

    
    def carregarModelo(self, caminhoArquivo):
        """
        Carrega o modelo de um arquivo especificado.

        Args:
            caminho_arquivo (str): Caminho completo para o arquivo do modelo.
        
        Returns:
            O modelo carregado.
        """
        if not os.path.exists(caminhoArquivo):
            raise FileNotFoundError(f"Arquivo não encontrado: {caminhoArquivo}")
        modelo = joblib.load(caminhoArquivo)
        print(f"Modelo carregado de: {caminhoArquivo}")
        return modelo