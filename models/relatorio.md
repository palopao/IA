# Relatório Comparativo de Modelos de Faringite

## Random Forest

### Importância dos Sintomas

| Sintoma | Importância |
| :--- | :--- |
| Temperatura_Corporal | 0.1649 |
| Tipo_Tosse | 0.1385 |
| Cefaleia | 0.1316 |
| Hipertrofia_Amigdalas | 0.1132 |
| Petequias_Palato | 0.1131 |
| Intensidade_Dor_Garganta | 0.0914 |
| Inicio_Subito | 0.0678 |
| Duracao_Sintomas_Dias | 0.0542 |
| Coriza | 0.0384 |
| Idade | 0.0300 |
| Halitose | 0.0209 |
| Mialgia | 0.0104 |
| Grau_Exsudato | 0.0100 |
| Conjuntivite | 0.0057 |
| Ganglios_Inchados | 0.0045 |
| Dor_Abdominal_Nauseas | 0.0029 |
| Adenopatia_Dolorosa | 0.0024 |

### Métricas de Desempenho

- **Precisão Global (Accuracy):** 0.9465

#### Relatório de Classificação Detalhado

| Classe | Precisão | Recall | F1-Score | Suporte |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 0.9368 | 0.9909 | 0.9631 | 10564 |
| 1 | 0.9749 | 0.8408 | 0.9029 | 4436 |
| **Global (Accuracy)** | | | **0.9465** | 15000 |
| macro avg | 0.9559 | 0.9159 | 0.9330 | 15000 |
| weighted avg | 0.9481 | 0.9465 | 0.9453 | 15000 |

---

## Decision Tree

### Importância dos Sintomas

| Sintoma | Importância |
| :--- | :--- |
| Inicio_Subito | 0.3997 |
| Hipertrofia_Amigdalas | 0.2021 |
| Intensidade_Dor_Garganta | 0.1468 |
| Duracao_Sintomas_Dias | 0.0887 |
| Cefaleia | 0.0771 |
| Temperatura_Corporal | 0.0276 |
| Tipo_Tosse | 0.0265 |
| Halitose | 0.0101 |
| Petequias_Palato | 0.0063 |
| Adenopatia_Dolorosa | 0.0058 |
| Coriza | 0.0037 |
| Idade | 0.0031 |
| Mialgia | 0.0018 |
| Conjuntivite | 0.0002 |
| Ganglios_Inchados | 0.0002 |
| Grau_Exsudato | 0.0002 |
| Dor_Abdominal_Nauseas | 0.0000 |

### Métricas de Desempenho

- **Precisão Global (Accuracy):** 0.9363

#### Relatório de Classificação Detalhado

| Classe | Precisão | Recall | F1-Score | Suporte |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 0.9337 | 0.9790 | 0.9558 | 10564 |
| 1 | 0.9434 | 0.8345 | 0.8856 | 4436 |
| **Global (Accuracy)** | | | **0.9363** | 15000 |
| macro avg | 0.9386 | 0.9068 | 0.9207 | 15000 |
| weighted avg | 0.9366 | 0.9363 | 0.9351 | 15000 |

---

## Classe Maioritária

### Métricas de Desempenho

- **Precisão Global (Accuracy):** 0.7043

#### Relatório de Classificação Detalhado

| Classe | Precisão | Recall | F1-Score | Suporte |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 0.7043 | 1.0000 | 0.8265 | 10564 |
| 1 | 0.0000 | 0.0000 | 0.0000 | 4436 |
| **Global (Accuracy)** | | | **0.7043** | 15000 |
| macro avg | 0.3521 | 0.5000 | 0.4132 | 15000 |
| weighted avg | 0.4960 | 0.7043 | 0.5821 | 15000 |

---

