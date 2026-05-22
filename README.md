# Diagnóstico AI: Faringite

Este projeto é um **Assistente de Inteligência Artificial** desenvolvido para auxiliar na classificação de faringite, distinguindo entre **Infeção Bacteriana (Streptococcus)** e **Infeção Viral (Virose)**. 

O sistema utiliza modelos de Machine Learning (Random Forest, Decision Tree e Classe Maioritária) treinados em dados sintéticos que simulam critérios clínicos reais (como as escalas de McIsaac/Centor e Brodsky).

**Apresentação:** ![presentation](presentation.pdf)

## Funcionalidades

*   **Interface Web:** Desenvolvida com Streamlit para fácil interação.
*   **Geração de Dados:** Script automático para gerar datasets baseados em probabilidades clínicas.
*   **Treino de Modelos:** Pipeline para treinar e comparar múltiplos algoritmos.
*   **Análise Técnica:** Visualização da importância das características e da estrutura da Árvore de Decisão.
*   **Relatórios:** Geração automática de relatórios de métricas (Precisão, Recall, F1-Score).

## Pré-requisitos

Antes de correr o projeto, certifica-te de que tens o **Python 3.8+** instalado. Além disso, para a visualização gráfica da Árvore de Decisão, é necessário ter o software **Graphviz** instalado no sistema.

## Instalação

1.  **Instalar as bibliotecas Python:**
    Executa o seguinte comando para instalar todas as dependências necessárias:

    ```bash
    pip install streamlit pandas scikit-learn joblib matplotlib seaborn graphviz numpy
    ```

2.  **Instalar o Graphviz (Software do Sistema):**
    *   **Windows:** Podes descarregar o instalador em [graphviz.org](https://graphviz.org/download/) e adicionar o diretório `bin` ao PATH do sistema.
    *   **Linux (Ubuntu/Debian):** `sudo apt-get install graphviz`
    *   **macOS:** `brew install graphviz`

## Como Correr

O projeto está estruturado para que o utilizador apenas precise de iniciar a aplicação principal. Se os modelos ou o dataset não existirem, a aplicação irá gerá-los automaticamente.

Para iniciar o assistente, executa:

```bash
python -m streamlit run app.py
```

## Estrutura do Projeto

*   `app.py`: O ficheiro principal que contém a interface Streamlit.
*   `data/generator.py`: Script responsável por gerar o dataset `sintomas_dataset.csv`.
*   `models/trainer.py`: Script que treina os modelos e gera os gráficos/relatórios técnicos.
*   `models/`: Pasta que armazena os modelos guardados (`.pkl`) e as imagens de análise.
*   `data/`: Pasta onde o dataset é armazenado.

## Aviso Legal

Este é um projeto **académico** (Proof of Concept). Os resultados gerados pela IA servem apenas para fins demonstrativos e **não substituem um diagnóstico médico profissional**.

---
*Desenvolvido para a unidade curricular de IA - 2º Semestre.*