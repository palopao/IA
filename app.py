import os
import subprocess
import streamlit as st
import pandas as pd
import joblib
import sys

# Configuração da página
st.set_page_config(page_title="Diagnóstico AI: Faringite", layout="centered")

@st.cache_data
def carregar_ficheiro_binario(caminho):
    """Lê e faz cache de ficheiros binários (imagens, datasets, etc) para evitar I/O repetitivo."""
    if os.path.exists(caminho):
        with open(caminho, "rb") as f:
            return f.read()
    return None

@st.cache_data
def carregar_ficheiro_texto(caminho):
    """Lê e faz cache de ficheiros de texto (relatórios MD)."""
    if os.path.exists(caminho):
        with open(caminho, "r", encoding="utf-8") as f:
            return f.read()
    return None

@st.cache_resource
def carregar_modelo(model_type):

    caminho_modelo_rf = 'models/modelo_faringite_rf.pkl'
    caminho_modelo_dt = 'models/modelo_faringite_dt.pkl'
    caminho_modelo_mc = 'models/modelo_faringite_mc.pkl'
    caminho_generator = 'data/sintomas_dataset.csv'

    if (
        not os.path.exists(caminho_modelo_rf)
        or not os.path.exists(caminho_modelo_dt)
        or not os.path.exists(caminho_modelo_mc)
        or not os.path.exists(caminho_generator)
    ):

        st.warning("Modelos ou dataset não encontrados.")

        try:

            # -------------------------
            # GERAR DATASET
            # -------------------------
            if not os.path.exists(caminho_generator):

                st.info("A gerar dataset...")

                resultado = subprocess.run(
                    [sys.executable, "data/generator.py"],
                    capture_output=True,
                    text=True
                )

                if resultado.returncode != 0:
                    st.error("generator.py falhou.")
                    return None

            # -------------------------
            # TREINAR MODELOS
            # -------------------------
            st.info("A treinar modelos...")

            resultado = subprocess.run(
                [sys.executable, "models/trainer.py"],
                capture_output=True,
                text=True
            )

            if resultado.returncode != 0:
                st.error("trainer.py falhou.")
                return None

            st.success("Modelos treinados!")

        except Exception as e:
            st.error(str(e))
            return None

    # -------------------------
    # CARREGAR MODELOS
    # -------------------------
    try:

        if model_type == 'random_forest':
            return joblib.load(caminho_modelo_rf)

        elif model_type == 'decision_tree':
            return joblib.load(caminho_modelo_dt)

        elif model_type == 'majority_class':
            return joblib.load(caminho_modelo_mc)

        else:
            st.error("Tipo inválido.")
            return None

    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None

st.title("Assistente IA: Classificação de Faringite")
st.markdown("Avaliação de probabilidade: **Infeção Bacteriana (Streptococcus)** vs **Infeção Viral (Virose)**.")

# Adicionar seleção de modelo
st.subheader("Seleção do Modelo de IA")
model_selection = st.radio(
    "Escolha o algoritmo de classificação:",
    ('Random Forest Classifier', 'Decision Tree Classifier', 'Classe Maioritária'),
    index=0, # Default para Random Forest
    help="Selecione o algoritmo de Machine Learning a ser utilizado para o diagnóstico."
)

if model_selection == 'Random Forest Classifier':
    selected_model_type = 'random_forest'
elif model_selection == 'Decision Tree Classifier':
    selected_model_type = 'decision_tree'
else:
    selected_model_type = 'majority_class'

modelo = carregar_modelo(selected_model_type)

with st.expander("Ver Análise Técnica do Modelo (Exclusivo Médico)"):
    st.subheader("Importância das Características no Diagnóstico")
    st.text("Este gráfico gerado pela nossa IA demonstra quais os sintomas que mais influenciaram a decisão do modelo de Machine Learning:")
    try:
        if (selected_model_type == 'random_forest'):
            st.markdown("Precisão: 94,65%")
            img_rf = carregar_ficheiro_binario("models/importancia_sintomas.png")
            if img_rf:
                st.image(img_rf, width='stretch')
        elif (selected_model_type == 'decision_tree'):
            st.markdown("Precisão: 93,63%")
            tab1, tab2 = st.tabs(["Importância dos Sintomas", "Estrutura da Árvore"])
            with tab1:
                img_dt = carregar_ficheiro_binario("models/importancia_sintomas_dt.png")
                if img_dt:
                    st.image(img_dt, width='stretch')
            with tab2:
                img_tree = carregar_ficheiro_binario("models/estrutura_arvore_dt.png")
                if img_tree:
                    st.image(img_tree, width='stretch')
                    
                    st.download_button(
                        label="📥 Descarregar Árvore em Alta Resolução (PNG)",
                        data=img_tree,
                        file_name="arvore_decisao_detalhada.png",
                        mime="image/png"
                    )
        elif (selected_model_type == 'majority_class'):
            st.markdown("Precisão: 70,43%")
            img_mc = carregar_ficheiro_binario("models/distribuicao_classes.png")
            if img_mc:
                st.image(img_mc, width='stretch')
            st.info("O modelo de Classe Maioritária baseia-se apenas na frequência das classes no dataset, ignorando os sintomas individuais para a sua previsão.")
    except:
        st.info("Gráfico de desempenho ainda não gerado pelo treinador.")

# Botão para re-gerar dados e treinar modelos
st.subheader("Manutenção do Modelo")
if st.button("Gerar Novos Dados e Treinar Modelos", help="Gera um novo dataset e treina ambos os modelos novamente."):
    st.info("A gerar o dataset e a treinar os modelos. Por favor, aguarde...")
    try:
        # Limpar o cache para garantir que os novos modelos e imagens são carregados do disco novamente
        st.cache_resource.clear()
        st.cache_data.clear()
        
        subprocess.run(
                    [sys.executable, "data/generator.py"],
                    capture_output=True,
                    text=True
                )
        subprocess.run(
                [sys.executable, "models/trainer.py"],
                capture_output=True,
                text=True
            )
        
        st.success("Dataset gerado e modelos treinados com sucesso! Por favor, atualize a página.")
        # Forçar o recarregamento do modelo após o treino
        modelo = carregar_modelo(selected_model_type)
    except subprocess.CalledProcessError as e:
        st.error(f"Erro ao re-gerar dados ou treinar modelos: {e}")
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado: {e}")

st.subheader("Dados do Paciente")
idade = st.number_input("Idade", min_value=1, max_value=100, value=25,
                        help="Idade do paciente em anos.")

st.subheader("1. Características da Dor e Início")
inicio_subito = st.checkbox("Início súbito e dor de garganta muito intensa ao engolir",
                            help="A dor de garganta apareceu de repente e é muito forte, especialmente ao engolir?")
intensidade_dor_garganta = st.slider("Intensidade da Dor de Garganta", min_value=1, max_value=10, value=0,
                                     help="Numa escala de 0 a 10, quão forte é a dor de garganta? (0 = nula, 10 = insuportável)")
duracao_sintomas_dias = st.number_input("Duração dos Sintomas (dias)", min_value=0, max_value=30, value=1,
                                        help="Há quantos dias os sintomas começaram?")

st.subheader("2. Avaliação Clínica (Sintomas Típicos)")
col1, col2 = st.columns(2)
with col1:
    temperatura_corporal = st.slider("Temperatura Corporal (ºC)", min_value=35.0, max_value=42.0, value=37.0, step=0.1,
                                     help="Qual a temperatura corporal mais alta registada? (Ex: 38.5ºC)")
    
    grau_exsudato_options = {
        "0 - Ausente": 0,
        "1 - Folicular (Pequenos pontos de pus isolados)": 1,
        "2 - Confluente (Placas de pus que se unem)": 2,
        "3 - Pseudomembranoso (Cobertura extensa das amígdalas)": 3
    }
    grau_exsudato_selection = st.radio("Grau de Exsudato (Pus nas Amígdalas)", list(grau_exsudato_options.keys()),
                                        index=0, help="Observa-se pus (pontos brancos ou placas) nas amígdalas?")
    grau_exsudato = grau_exsudato_options[grau_exsudato_selection]

    hipertrofia_amigdalas_options = {
        "0 - Ausente (Amígdalas não visíveis ou pequenas)": 0,
        "1 - Leve (Amígdalas ocupam <25% do espaço entre os pilares)": 1,
        "2 - Moderada (Amígdalas ocupam 25-50% do espaço)": 2,
        "3 - Grande (Amígdalas ocupam 50-75% do espaço)": 3,
        "4 - Obstrutiva (Amígdalas ocupam >75% do espaço, quase se tocam)": 4
    }
    hipertrofia_amigdalas_selection = st.radio("Grau de Hipertrofia das Amígdalas", list(hipertrofia_amigdalas_options.keys()),
                                                index=0, help="Qual o tamanho das amígdalas? (Baseado na Escala de Brodsky)")
    hipertrofia_amigdalas = hipertrofia_amigdalas_options[hipertrofia_amigdalas_selection]

with col2:
    ganglios_inchados = st.checkbox("Aumento dos gânglios no pescoço (ínguas)")
    adenopatia_dolorosa = st.checkbox("Gânglios dolorosos ao toque",
                                      help="Os gânglios inchados no pescoço doem quando são tocados?")
    dor_abdominal = st.checkbox("Dor abdominal, náuseas ou vómitos")
    petequias_palato = st.checkbox("Petéquias no Palato (pontos vermelhos no céu da boca)",
                                   help="Observa pequenos pontos vermelhos (como picadas) no céu da boca?")
    halitose = st.checkbox("Mau hálito (Halitose)",
                           help="O paciente apresenta mau hálito persistente?")

st.subheader("3. Sintomas de Constipação (Sintomas Virais)")
st.info("Nota: A presença destes sintomas costuma indicar causa viral.")
col3, col4 = st.columns(2)
with col3:
    tipo_tosse_options = {
        "0 - Ausente": 0,
        "1 - Seca (sem expetoração)": 1,
        "2 - Produtiva (com expetoração)": 2
    }
    tipo_tosse_selection = st.radio("Tipo de Tosse", list(tipo_tosse_options.keys()), index=0,
                                    help="O paciente tem tosse? Se sim, qual o tipo?")
    tipo_tosse = tipo_tosse_options[tipo_tosse_selection]
    coriza = st.checkbox("Coriza (nariz a pingar)")
with col4:
    conjuntivite = st.checkbox("Conjuntivite / Olhos vermelhos")
    mialgia = st.checkbox("Mialgia (Dores no corpo)",
                          help="O paciente sente dores musculares generalizadas?")
    cefaleia = st.slider("Intensidade da Cefaleia (Dor de cabeça)", min_value=0, max_value=10, value=0,
                         help="Numa escala de 1 a 10, quão forte é a dor de cabeça? (0 = nula, 10 = insuportável)")

if st.button("Analisar Sintomas", type="primary"):
    if modelo:
        dados = pd.DataFrame({
            'Idade': [idade],
            'Inicio_Subito': [1 if inicio_subito else 0],
            'Temperatura_Corporal': [temperatura_corporal],
            'Intensidade_Dor_Garganta': [intensidade_dor_garganta],
            'Duracao_Sintomas_Dias': [duracao_sintomas_dias],
            'Hipertrofia_Amigdalas': [hipertrofia_amigdalas],
            'Grau_Exsudato': [grau_exsudato],
            'Ganglios_Inchados': [1 if ganglios_inchados else 0],
            'Dor_Abdominal_Nauseas': [1 if dor_abdominal else 0],
            'Tipo_Tosse': [tipo_tosse],
            'Coriza': [1 if coriza else 0],
            'Conjuntivite': [1 if conjuntivite else 0],
            'Petequias_Palato': [1 if petequias_palato else 0],
            'Mialgia': [1 if mialgia else 0],
            'Cefaleia': [cefaleia],
            'Halitose': [1 if halitose else 0],
            'Adenopatia_Dolorosa': [1 if adenopatia_dolorosa else 0]
        })
        
        probabilidades = modelo.predict_proba(dados)[0]
        previsao = modelo.predict(dados)[0]
        prob = max(probabilidades[0], probabilidades[1])*100

        if prob >= 85:
            risco = "Muito Elevado"
        elif prob >= 70:
            risco = "Elevado"
        elif prob >= 55:
            risco = "Moderado"
        else:
            risco = "Reduzido"
        
        st.divider()
        
        if previsao == 1:
            st.error(f"### Risco {risco} de Infeção Bacteriana ({prob:.1f}%)")
            st.markdown("""
            **Análise:** O quadro clínico sugere uma faringite estreptocócica.
            
            **Próximos Passos recomendados:**
            * Consultar um médico para avaliação presencial e eventual zaragatoa.
            * **Tratamento provável:** Antibióticos (apenas com prescrição médica) para evitar complicações graves.
            """)
        else:
            st.success(f"### Risco {risco} de Infeção Viral ({prob:.1f}%)")
            st.markdown("""
            **Análise:** O quadro clínico sugere uma virose comum.
            
            **Próximos Passos recomendados:**
            * **Tratamento provável:** Sintomáticos (analgésicos, anti-inflamatórios, repouso e hidratação).
            * Monitorizar evolução da febre.
            """)
            
st.caption("Aviso: Este é um projeto académico (Proof of Concept). Os resultados não substituem um diagnóstico médico profissional.")

with st.expander("Ficheiros e Dados do Projeto"):
    st.subheader("1. Dataset de Treino (Amostra)")
    try:
        csv_bytes = carregar_ficheiro_binario("data/sintomas_dataset.csv")
        if csv_bytes:
            import io
            df_preview = pd.read_csv(io.BytesIO(csv_bytes))
            st.dataframe(df_preview.head(50))
            st.download_button(
                label="📥 Descarregar Dataset Completo (CSV)",
                data=csv_bytes,
                file_name="sintomas_dataset.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Erro ao carregar dataset: {e}")

    st.divider()

    st.subheader("2. Relatório Técnico Completo")
    try:
        conteudo_relatorio = carregar_ficheiro_texto("models/relatorio.md")
        if conteudo_relatorio:
            st.markdown(conteudo_relatorio)
            st.download_button(
                label="📥 Descarregar Relatório (Markdown)",
                data=conteudo_relatorio,
                file_name="relatorio_tecnico.md",
                mime="text/markdown"
            )
    except Exception as e:
        st.error(f"Erro ao carregar relatório: {e}")