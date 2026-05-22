import pandas as pd
import numpy as np

def gerar_dados(num_samples=50000):
    np.random.seed(42)
    
    # Gerar idades (dar algum peso à faixa 5-15 anos)
    idades = np.random.randint(3, 65, num_samples)
    
    # Target: 1 = Bacteriana (Streptococcus), 0 = Viral
    # Distribuir as doenças (aprox. 30% bacteriana, 70% viral para ser realista)
    # Ajustar a probabilidade inicial com base na prevalência por idade (critérios de McIsaac/Centor)
    infeccao_bacteriana = np.zeros(num_samples, dtype=int)
    for i in range(num_samples):
        age = idades[i]
        if 3 <= age <= 14: # McIsaac: +1 ponto, maior chance de bacteriana
            infeccao_bacteriana[i] = np.random.choice([0, 1], p=[0.5, 0.5]) # 50% chance
        elif 15 <= age <= 44: # McIsaac: 0 pontos, chance moderada
            infeccao_bacteriana[i] = np.random.choice([0, 1], p=[0.7, 0.3]) # 30% chance
        else: # <3 ou >=45 (McIsaac: -1 ponto para >45), bacteriana é muito rara
            infeccao_bacteriana[i] = np.random.choice([0, 1], p=[0.95, 0.05]) # 5% chance

    # Inicializar arrays de sintomas complexos
    inicio_subito = np.zeros(num_samples, dtype=int)
    temperatura_corporal = np.zeros(num_samples) # Contínua
    intensidade_dor_garganta = np.zeros(num_samples, dtype=int) # Escala 0-10
    duracao_sintomas_dias = np.zeros(num_samples, dtype=int) # Número de dias
    hipertrofia_amigdalas = np.zeros(num_samples, dtype=int) # Escala 0-4 (Brodsky)
    grau_exsudato = np.zeros(num_samples, dtype=int) # Escala 0-3
    ganglios_inchados = np.zeros(num_samples, dtype=int)
    dor_abdominal = np.zeros(num_samples, dtype=int)
    tipo_tosse = np.zeros(num_samples, dtype=int) # 0=Ausente, 1=Seca, 2=Produtiva
    coriza = np.zeros(num_samples, dtype=int)
    conjuntivite = np.zeros(num_samples, dtype=int)
    petequias_palato = np.zeros(num_samples, dtype=int)
    mialgia = np.zeros(num_samples, dtype=int)
    cefaleia = np.zeros(num_samples, dtype=int) # Escala 0-10
    halitose = np.zeros(num_samples, dtype=int)
    adenopatia_dolorosa = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        age = idades[i]
        is_bacterial = infeccao_bacteriana[i]

        if is_bacterial == 1: # Se for Bacteriana
            inicio_subito[i] = np.random.choice([0, 1], p=[0.3, 0.7]) # Menos determinístico
            temperatura_corporal[i] = np.random.uniform(37.2, 40.5) # Mais overlap na base da febre
            intensidade_dor_garganta[i] = np.random.randint(4, 11) # Mais overlap na dor
            duracao_sintomas_dias[i] = np.random.randint(1, 8) # Aumentada variabilidade (era 2-4)
            hipertrofia_amigdalas[i] = np.random.choice([1, 2, 3, 4], p=[0.2, 0.3, 0.3, 0.2]) 
            grau_exsudato[i] = np.random.choice([0, 1, 2, 3], p=[0.25, 0.3, 0.3, 0.15]) # Mais casos bacterianos sem pus
            ganglios_inchados[i] = np.random.choice([0, 1], p=[0.35, 0.65]) # Redução da força deste sinal
            
            # Cluster: Adenopatia dolorosa só faz sentido se os gânglios estiverem inchados
            adenopatia_dolorosa[i] = np.random.choice([0, 1], p=[0.6, 0.4]) if ganglios_inchados[i] == 1 else 0 
            
            # Dor abdominal é mais comum se for criança (5-15 anos)
            if 5 <= age <= 15:
                dor_abdominal[i] = np.random.choice([0, 1], p=[0.5, 0.5]) # 50% chance em crianças
            else:
                dor_abdominal[i] = np.random.choice([0, 1], p=[0.8, 0.2])
                
            # Sintomas virais são raros aqui
            tipo_tosse[i] = np.random.choice([0, 1], p=[0.92, 0.08]) # Reduzida para ser um preditor negativo mais forte
            coriza[i] = np.random.choice([0, 1], p=[0.90, 0.10]) 
            conjuntivite[i] = np.random.choice([0, 1], p=[0.93, 0.07]) 

            petequias_palato[i] = np.random.choice([0, 1], p=[0.3, 0.7]) # reduzido
            mialgia[i] = np.random.choice([0, 1], p=[0.8, 0.2]) # Menos comum
            cefaleia[i] = np.random.randint(3, 9) 
            
            # Cluster: Halitose fortemente ligada ao pus (Exsudato)
            halitose[i] = np.random.choice([0, 1], p=[0.1, 0.9]) if grau_exsudato[i] >= 2 else np.random.choice([0, 1], p=[0.6, 0.4])

            # Aplicar correlações para bacteriana
            if petequias_palato[i] == 1: 
                # Petéquias aumentam chance de outros sintomas, mas não garantem exsudato máximo
                temperatura_corporal[i] = max(temperatura_corporal[i], np.random.uniform(38.0, 40.0))
                if np.random.random() < 0.5: # Reduzida a correlação forçada
                    adenopatia_dolorosa[i] = np.random.choice([0, 1], p=[0.4, 0.6])
                    ganglios_inchados[i] = np.random.choice([0, 1], p=[0.2, 0.8])
                # Reduzimos a probabilidade de sintomas virais, mas não zeramos
                tipo_tosse[i] = np.random.choice([0, 1], p=[0.9, 0.1])

            # Se pus elevado e bacteriano, a febre tende a ser maior
            if grau_exsudato[i] >= 2 and is_bacterial == 1 and np.random.random() < 0.8:
                if temperatura_corporal[i] < 38.2:
                    temperatura_corporal[i] = np.random.uniform(38.5, 40.2)
            
        else: # Se for Viral
            inicio_subito[i] = np.random.choice([0, 1], p=[0.9, 0.1]) # Início gradual muito mais comum
            temperatura_corporal[i] = np.random.uniform(36.0, 39.2) # Aumentado o topo da febre viral
            intensidade_dor_garganta[i] = np.random.randint(0, 8) # Aumentado o overlap (era 0-7)
            duracao_sintomas_dias[i] = np.random.randint(2, 14) # Aumentado o overlap (era 3-11)
            hipertrofia_amigdalas[i] = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2]) # Hipertrofia leve ou ausente
            grau_exsudato[i] = np.random.choice([0, 1, 2], p=[0.55, 0.30, 0.15]) # Mais virais com pus
            ganglios_inchados[i] = np.random.choice([0, 1], p=[0.7, 0.3]) 
            adenopatia_dolorosa[i] = np.random.choice([0, 1], p=[0.8, 0.2]) if ganglios_inchados[i] == 1 else 0 # Aumentado para 20% no vírus
            dor_abdominal[i] = np.random.choice([0, 1], p=[0.95, 0.05]) # Dor abdominal muito rara
            
            # Sintomas virais são comuns
            tipo_tosse[i] = np.random.choice([0, 1, 2], p=[0.2, 0.4, 0.4]) 
            coriza[i] = np.random.choice([0, 1], p=[0.3, 0.7]) # Coriza muito comum
            conjuntivite[i] = np.random.choice([0, 1], p=[0.6, 0.4]) # Conjuntivite mais comum

            petequias_palato[i] = np.random.choice([0, 1], p=[0.94, 0.06]) # Aumentado: 6% das virais podem ter
            mialgia[i] = np.random.choice([0, 1], p=[0.3, 0.7]) # Mais comum
            cefaleia[i] = np.random.randint(0, 6) # Dor de cabeça leve a moderada
            halitose[i] = np.random.choice([0, 1], p=[0.9, 0.1]) # Menos comum

            # Evitar regras 100% rígidas para forçar a árvore a explorar outros sintomas
            if (coriza[i] == 1 or tipo_tosse[i] > 0) and np.random.random() < 0.05:
                infeccao_bacteriana[i] = 1 # Rara faringite bacteriana com tosse/coriza

    # Adicionar 2% de ruído puro (erros de diagnóstico/casos atípicos)
    # Isto impede que qualquer modelo atinja 100% de precisão com uma única regra
    indices_ruido = np.random.choice(num_samples, size=int(num_samples * 0.02), replace=False)
    for idx in indices_ruido:
        infeccao_bacteriana[idx] = 1 - infeccao_bacteriana[idx]

    df = pd.DataFrame({
        'Idade': idades,
        'Inicio_Subito': inicio_subito,
        'Temperatura_Corporal': np.round(temperatura_corporal, 1), # Arredondar para 1 casa decimal
        'Intensidade_Dor_Garganta': intensidade_dor_garganta,
        'Duracao_Sintomas_Dias': duracao_sintomas_dias,
        'Hipertrofia_Amigdalas': hipertrofia_amigdalas,
        'Grau_Exsudato': grau_exsudato,
        'Ganglios_Inchados': ganglios_inchados,
        'Dor_Abdominal_Nauseas': dor_abdominal,
        'Tipo_Tosse': tipo_tosse,
        'Coriza': coriza,
        'Conjuntivite': conjuntivite,
        'Petequias_Palato': petequias_palato,
        'Mialgia': mialgia,
        'Cefaleia': cefaleia,
        'Halitose': halitose,
        'Adenopatia_Dolorosa': adenopatia_dolorosa,
        'Infeccao_Bacteriana': infeccao_bacteriana
    })
    
    df.to_csv('data/sintomas_dataset.csv', index=False)
    print("Dataset completo gerado!")

if __name__ == "__main__":
    gerar_dados()