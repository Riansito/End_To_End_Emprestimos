# Projeto: Previsão de Inadimplência de Empréstimos

Este projeto foi desenvolvido com o objetivo de prever inadimplências em empréstimos com base em um conjunto de dados fornecido. A abordagem é end-to-end, contemplando desde a análise exploratória até a modelagem preditiva.

## Tecnologias Utilizadas

* **Pandas**: Manipulação e análise de dados.
* **NumPy**: Suporte para cálculos numéricos.
* **Seaborn**: Criação de visualizações estatísticas.
* **Matplotlib**: Construção de gráficos e visualizações.
* **Scikit-learn**: Modelagem preditiva e avaliação de modelos.

## Estrutura do Projeto

O projeto está estruturado da seguinte forma:

1. **Data Preparation**: Pré-processamento e limpeza dos dados, incluindo o tratamento de valores ausentes e codificação de variáveis categóricas.
2. **Exploratory Data Analysis (EDA)**: Análise exploratória para identificar correlações e padrões relevantes no dataset.
3. **Feature Engineering**: Seleção e criação de variáveis importantes para melhorar o desempenho do modelo.
4. **Modelagem**: Treinamento de modelos supervisionados, como árvores de decisão e XGBoost, para prever inadimplência.
5. **Avaliação**: Análise do desempenho do modelo utilizando métricas como acurácia, precisão, recall e F1-score.

## Exemplos de Visualizações

As visualizações ajudam a entender os resultados e o desempenho do modelo. Abaixo, apresentamos dois gráficos importantes:

<div style="display: flex; justify-content: space-around;">
  <div style="text-align: center;">
    <img src="Imagens/Captura%20de%20tela%202025-06-03%20151646.png" alt="Curva ROC" width="400">
    <p><b>Curva ROC:</b> Avalia o desempenho do modelo, mostrando a relação entre a taxa de verdadeiros positivos (TPR) e a taxa de falsos positivos (FPR). Uma curva mais próxima do canto superior esquerdo indica um modelo mais eficiente.</p>
  </div>
  <div style="text-align: center;">
    <img src="Imagens/Captura%20de%20tela%202025-06-03%20151632.png" alt="Matriz de Confusão" width="400">
    <p><b>Matriz de Confusão:</b> Exibe os resultados de classificação do modelo, detalhando verdadeiros positivos, falsos positivos, verdadeiros negativos e falsos negativos. Útil para identificar onde o modelo comete erros.</p>
  </div>
</div>

## Objetivo do Modelo

A meta principal do projeto foi criar um modelo preditivo robusto e eficiente, capaz de identificar clientes com maior probabilidade de inadimplência. O foco foi equilibrar precisão e interpretabilidade para apoiar decisões estratégicas na concessão de crédito.

## Como Executar o Projeto

1. Clone este repositório:

   ```bash
   git clone https://github.com/Riansito/End_To_End_Emprestimos.git
   ```
2. Instale as dependências necessárias (recomenda-se o uso de um ambiente virtual):

   ```bash
   pip install -r requirements.txt
   ```
3. Execute o script principal:

   ```bash
   python main.py
   ```

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests com melhorias ou sugestões. Vamos construir juntos!



## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests com melhorias ou sugestões.


