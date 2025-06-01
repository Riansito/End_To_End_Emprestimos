import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize

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
        
    @staticmethod
    def preencherValoresVaziosCreditScore(linha):
        """
        Trata os valores faltantes da coluna CreditScore com base no Loan Status
        """
        if pd.isnull(linha['Credit Score']):
            if linha['Loan Status'] == 'Fully Paid':
                return 716
            elif linha['Loan Status'] == 'Charged Off':
                return 2402
        return linha['Credit Score']
    

    def aplicarWinsorizacao(self, df, colunas=None, limites=(0.05, 0.05)):
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
    
    

    

class Graficos:
    def __init__(self):
        pass  

    # Método de instância (sem @staticmethod)
    def graficoAnaliseOutlier(self, dfAnaliseOutlier):
        """
        Gera boxplots para análise de outliers de colunas numéricas
        
        Parâmetros:
        - dfAnaliseOutlier: DataFrame com os dados a serem analisados
        """
        colunasNumericas = dfAnaliseOutlier.select_dtypes(include="number").columns
        
        numeroLinhas = 4  

        plt.figure(figsize=[15, 5 * numeroLinhas])
        for i, coluna in enumerate(colunasNumericas):
            plt.subplot(numeroLinhas, 3, i+1)  # Corrigido o layout
            sns.boxplot(y=dfAnaliseOutlier[coluna])
        plt.tight_layout()
        plt.show()

    # Método de instância (sem @staticmethod)
    def graficoAnaliseOutlierPorLoanStatus(self, dfAnaliseOutlier):
        """
        Gera boxplots comparando colunas numéricas por Loan Status
        
        Parâmetros:
        - dfAnaliseOutlier: DataFrame com os dados a serem analisados
        """
        colunasNumericas = dfAnaliseOutlier.select_dtypes(include="number").drop(["Current Loan Amount", "Credit Score"], axis=1).columns
        
        numeroLinhas = 4  

        plt.figure(figsize=[15, 5 * numeroLinhas])
        for i, coluna in enumerate(colunasNumericas):
            plt.subplot(numeroLinhas, 3, i+1)  # Corrigido o layout
            sns.boxplot(x=dfAnaliseOutlier["Loan Status"], y=dfAnaliseOutlier[coluna])
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