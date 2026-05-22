import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import graphviz
import joblib
from sklearn.tree import export_graphviz

def plot_feature_importance_rf(feature_names, importances, output_path='models/importancia_sintomas.png'):
    """
    Gera e guarda o gráfico de importância de características para o modelo Random Forest.
    """
    importancia_df = pd.DataFrame({
        'Sintoma': feature_names,
        'Importancia': importances
    }).sort_values(by='Importancia', ascending=True)

    plt.figure(figsize=(10, 6))
    importancia_df.plot(
        kind='barh',
        x='Sintoma',
        y='Importancia',
        legend=False,
        figsize=(10, 6)
    )
    plt.title('Importância dos Sintomas no Modelo (Random Forest)')
    plt.xlabel('Importância')
    plt.ylabel('Sintomas')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de importância dos sintomas (Random Forest) guardado em {output_path}")

def plot_feature_importance_dt(feature_names, importances, output_path='models/importancia_sintomas_dt.png'):
    """
    Gera e guarda o gráfico de importância de características para o modelo Decision Tree.
    """
    importancia_dt_df = pd.DataFrame({
        'Sintoma': feature_names,
        'Importancia': importances
    }).sort_values(by='Importancia', ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(importancia_dt_df['Sintoma'], importancia_dt_df['Importancia'], color='seagreen')
    plt.title('Importância dos Sintomas (Decision Tree)')
    plt.xlabel('Importância')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Gráfico de importância (DT) guardado em {output_path}")

def plot_decision_tree_structure(model, feature_names, class_names, output_dir='models', filename='estrutura_arvore_dt'):
    """
    Gera e guarda a visualização da estrutura da Decision Tree.
    """
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        special_characters=True
    )
    dot_data = dot_data.replace(
        'digraph Tree {',
        '''digraph Tree {
        graph [
            rankdir=TB,
            splines=true,
            overlap=false,
            nodesep=0.3,
            ranksep=0.6,
        ];
        node [
            fontsize=9,
            margin="0.15,0.1"
        ];
        edge [
            penwidth=1.0,
            arrowsize=0.6
        ];
        '''
    )
    graph = graphviz.Source(dot_data)
    graph.engine = "twopi"
    graph.format = "png"
    output_path = graph.render(filename=filename, directory=output_dir, cleanup=True)
    print(f"Árvore guardada em: {output_path}")

def plot_class_distribution(y_data, output_path='models/distribuicao_classes.png'):
    """
    Gera e guarda o gráfico de distribuição de classes.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y_data, hue=y_data, palette='viridis', legend=False)
    plt.title('Distribuição de Classes no Dataset (Base da Classe Maioritária)')
    plt.xlabel('Tipo de Infeção (0: Viral, 1: Bacteriana)')
    plt.ylabel('Número de Casos')
    plt.xticks([0, 1], ['Viral', 'Bacteriana'])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Gráfico de distribuição de classes guardado em {output_path}")

if __name__ == "__main__":
    print("Gerando gráficos a partir dos modelos e dados salvos...")

    # 1. Carregar o dataset para obter os nomes das características e a distribuição das classes
    try:
        df = pd.read_csv('data/sintomas_dataset.csv')
        X_columns = df.drop('Infeccao_Bacteriana', axis=1).columns.tolist()
        y_data = df['Infeccao_Bacteriana']
        class_names = ['Viral', 'Bacteriana'] # Assumindo que são consistentes
    except FileNotFoundError:
        print("Erro: 'data/sintomas_dataset.csv' não encontrado. Certifique-se de que o dataset foi gerado.")
        exit()
    except Exception as e:
        print(f"Erro ao carregar o dataset: {e}")
        exit()

    # 2. Carregar os modelos treinados
    modelo_rf = None
    modelo_dt = None

    try:
        modelo_rf = joblib.load('models/modelo_faringite_rf.pkl')
        print("Modelo Random Forest carregado.")
    except FileNotFoundError:
        print("Aviso: 'models/modelo_faringite_rf.pkl' não encontrado. O gráfico de importância do RF não será gerado.")
    except Exception as e:
        print(f"Erro ao carregar modelo Random Forest: {e}")

    try:
        modelo_dt = joblib.load('models/modelo_faringite_dt.pkl')
        print("Modelo Decision Tree carregado.")
    except FileNotFoundError:
        print("Aviso: 'models/modelo_faringite_dt.pkl' não encontrado. Os gráficos da Decision Tree não serão gerados.")
    except Exception as e:
        print(f"Erro ao carregar modelo Decision Tree: {e}")

    # 3. Gerar gráficos se os modelos estiverem disponíveis
    if modelo_rf is not None:
        plot_feature_importance_rf(X_columns, modelo_rf.feature_importances_)
    if modelo_dt is not None:
        plot_feature_importance_dt(X_columns, modelo_dt.feature_importances_)
        plot_decision_tree_structure(modelo_dt, feature_names=X_columns, class_names=class_names)
    plot_class_distribution(y_data) # Este gráfico usa apenas os dados brutos, então pode ser gerado se o df for carregado

    print("Geração de gráficos concluída.")