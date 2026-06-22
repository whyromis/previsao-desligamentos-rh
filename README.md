# 🎯 Radar Preditivo de RH: Previsão de Desligamentos com Inteligência Artificial

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![XGBoost](https://img.shields.io/badge/Machine%20Learning-XGBoost-green)
![Status](https://img.shields.io/badge/Status-Em%20Produ%C3%A7%C3%A3o-success)

🔗 **[Acesse a aplicação rodando em produção aqui]** *https://previsao-desligamentos-rh-wug8bhfs5xjmzad4reezdh.streamlit.app/*

## 📖 O Problema de Negócio
A alta rotatividade (turnover) de estagiários é um desafio clássico para o setor de Recursos Humanos. A reposição de talentos exige tempo para recrutamento, seleção e onboarding. Quando o RH atua de forma reativa (só começa a procurar após o desligamento), as áreas de negócio sofrem com a queda de produtividade.

**O objetivo deste projeto foi mudar a postura do RH de "Reativa" para "Preditiva".**

## 💡 A Solução
Desenvolvi uma aplicação web interativa onde os analistas de RH podem fazer o upload de sua base de dados atualizada. O sistema processa o histórico, treina um modelo de Machine Learning em tempo real e gera um **Radar de Reposição**: um plano de ação indicando exatamente quantos desligamentos são esperados para os próximos meses e em quais setores eles devem ocorrer.

### ✨ Funcionalidades Principais:
- **Upload Dinâmico:** Aceita bases de dados em Excel (`.xlsx`) e `.csv`, tratando anomalias, dados faltantes e formatos de data divergentes automaticamente.
- **Previsão Macro (Empresa):** Gráfico de séries temporais mostrando a tendência histórica e a curva de previsão futura de desligamentos totais.
- **Radar Micro (Setor a Setor):** Tabela acionável que filtra o "ruído" estatístico e consolida o risco difuso, entregando para o recrutamento exatamente quantas vagas abrirão por área.
- **Exportação de Dados:** Geração de um relatório em CSV com as previsões prontas para integrar com ferramentas de BI.

---

## 🛠️ Arquitetura Técnica e Machine Learning

Para resolver este problema de Série Temporal, optei por não usar modelos clássicos como ARIMA, mas sim o **XGBoost Regressor**, modelando o problema como uma previsão de contagens.

### 1. Feature Engineering (A Memória do Modelo)
Para o XGBoost entender o tempo, criei variáveis temporais a partir do histórico de cada setor:
*   `lag_1` e `lag_2`: Número de desligamentos nos últimos 1 e 2 meses.
*   `media_movel_3m`: A tendência suavizada do trimestre anterior.
*   `mes_do_ano`: Captura a sazonalidade (ex: períodos típicos de fim de contrato ou formatura).

### 2. Modelagem (A Escolha do XGBoost)
*   **Objective de Poisson:** Como não existe "meia pessoa" demitida, modelei o XGBoost com `objective='count:poisson'`. Essa função matemática é ideal para prever eventos raros e de contagem inteira (como acidentes, falhas de máquina ou, neste caso, desligamentos).

### 3. Regra de Negócio (O Paradoxo do Risco Difuso)
Modelos geram probabilidades fracionadas (ex: 0.08 de chance de saída). Em vez de poluir a visão do RH com dezenas de setores com risco baixo por mês, apliquei uma **camada de lógica de negócio**: o aplicativo acumula o risco de cada setor ao longo do horizonte previsto e só alerta o RH quando o somatório estatístico ultrapassa 0.5 (indicando uma forte probabilidade de ao menos 1 vaga real). 

---

## 💻 Stack de Tecnologias Usadas
*   **Linguagem:** Python
*   **Manipulação de Dados:** Pandas, NumPy
*   **Machine Learning:** XGBoost, Scikit-Learn
*   **Visualização:** Matplotlib
*   **Frontend e Web App:** Streamlit
*   **Deploy (MLOps):** Streamlit Community Cloud (via GitHub)

---

## 🚀 Como executar o projeto localmente

Caso queira rodar o projeto na sua própria máquina, siga os passos:

1. Clone o repositório:
```bash
git clone https://github.com/whyromis/previsao-desligamentos-rh.git
```

2. Crie um ambiente virtual e ative-o:
```bash
python -m venv venv
# No Windows:
venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Execute a aplicação:
```bash
streamlit run app.py
```

---
*Desenvolvido com o foco em transformar dados brutos em decisões estratégicas.*
