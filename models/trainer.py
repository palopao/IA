import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def treinar_modelo():
    df = pd.read_csv('data/sintomas_dataset.csv')
    
    X = df.drop('Infeccao_Bacteriana', axis=1)
    y = df['Infeccao_Bacteriana']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # --- Treino do RandomForestClassifier ---
    # Aumentamos n_estimators para maior estabilidade
    # Aumentamos max_depth para permitir que o modelo aprenda relações mais complexas
    # class_weight='balanced' ajuda se o dataset for desequilibrado (ex: muito mais virais que bacterianas)
    # Adicionamos min_samples_leaf para evitar que a árvore foque em casos isolados (ruído)
    modelo_rf = RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_leaf=5, class_weight='balanced', random_state=42)
    modelo_rf.fit(X_train, y_train)
    
    # --- Treino do DecisionTreeClassifier ---
    # Reduzimos drasticamente o max_depth. Uma árvore sozinha não deve ser tão profunda para ser interpretável.
    # Adicionamos max_features='sqrt' para impedir que a árvore use sempre o "melhor" sintoma em todos os nós,
    # forçando-a a encontrar caminhos alternativos e distribuindo melhor a importância.
    modelo_dt = DecisionTreeClassifier(max_depth=10, min_samples_leaf=40, max_features='sqrt', class_weight='balanced', random_state=42)
    modelo_dt.fit(X_train, y_train)

    # --- Treino do DummyClassifier (Classe Maioritária) ---
    modelo_mc = DummyClassifier(strategy='most_frequent')
    modelo_mc.fit(X_train, y_train)
    
    # --- Geração do Relatório Comparativo ---
    
    modelos_para_relatorio = [
        ("Random Forest", modelo_rf),
        ("Decision Tree", modelo_dt),
        ("Classe Maioritária", modelo_mc)
    ]

    with open('models/relatorio.md', 'w', encoding='utf-8') as f:
        f.write("# Relatório Comparativo de Modelos de Faringite\n\n")

        for nome, modelo in modelos_para_relatorio:
            f.write(f"## {nome}\n\n")
            
            # Previsões e métricas
            y_pred = modelo.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            # Se o modelo tiver importância de características (RF e DT)
            if hasattr(modelo, 'feature_importances_'):
                f.write("### Importância dos Sintomas\n\n")
                f.write("| Sintoma | Importância |\n")
                f.write("| :--- | :--- |\n")
                importancia_tmp = pd.DataFrame({
                    'Sintoma': X.columns,
                    'Importancia': modelo.feature_importances_
                }).sort_values(by='Importancia', ascending=False)
                
                for _, row in importancia_tmp.iterrows():
                    f.write(f"| {row['Sintoma']} | {row['Importancia']:.4f} |\n")
                f.write("\n")

            f.write(f"### Métricas de Desempenho\n\n")
            f.write(f"- **Precisão Global (Accuracy):** {acc:.4f}\n\n")
            f.write("#### Relatório de Classificação Detalhado\n\n")
            f.write("| Classe | Precisão | Recall | F1-Score | Suporte |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            
            for label, metrics in report.items():
                if label == 'accuracy':
                    f.write(f"| **Global (Accuracy)** | | | **{metrics:.4f}** | {int(report['macro avg']['support'])} |\n")
                else:
                    f.write(f"| {label} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1-score']:.4f} | {int(metrics['support'])} |\n")
            f.write("\n---\n\n")

    print("Relatório comparativo guardado em models/relatorio.md")
    
    # Guardar ambos os modelos
    joblib.dump(modelo_rf, 'models/modelo_faringite_rf.pkl')
    print("Modelo Random Forest guardado em models/modelo_faringite_rf.pkl")
    
    joblib.dump(modelo_dt, 'models/modelo_faringite_dt.pkl')
    print("Modelo Decision Tree guardado em models/modelo_faringite_dt.pkl")

    joblib.dump(modelo_mc, 'models/modelo_faringite_mc.pkl')
    print("Modelo Classe Maioritária guardado em models/modelo_faringite_mc.pkl")

if __name__ == "__main__":
    treinar_modelo()