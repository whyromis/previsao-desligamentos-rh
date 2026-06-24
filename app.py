import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ---------------------------------------------------------
st.set_page_config(page_title="Predição de Desligamentos - RH", page_icon="👥", layout="wide")

st.title("📊 Radar de Reposição de Estagiários")
st.markdown("""
Bem-vindo ao portal preditivo do RH. Faça o upload da sua base de estagiários atualizada e 
nossa Inteligência Artificial calculará a previsão de desligamentos para os próximos meses, setor por setor.
""")

# ---------------------------------------------------------
# BARRA LATERAL (CONTROLES DO USUÁRIO)
# ---------------------------------------------------------
st.sidebar.header("⚙️ Configurações")
arquivo_upload = st.sidebar.file_uploader("1. Faça upload da base (Excel ou CSV)", type=["xlsx", "csv"])
meses_futuro = st.sidebar.slider("2. Meses para prever:", min_value=1, max_value=12, value=6)

# ---------------------------------------------------------
# FUNÇÕES CORE (COM CACHE PARA PERFORMANCE)
# ---------------------------------------------------------
@st.cache_data
@st.cache_data
def processar_dados(dfnovo):
    # 1. Padroniza colunas
    dfnovo.columns = [col.strip().upper() for col in dfnovo.columns]
    
    # 2. Correção de Data e Anomalias (BLINDADO)
    if 'MÊS REFERÊNCIA' in dfnovo.columns:
        # FORÇA a conversão para data, ignorando o que o Pandas acha que é.
        # errors='coerce' transforma textos inválidos (ex: linhas em branco) em NaT (Not a Time)
        dfnovo['MÊS REFERÊNCIA'] = pd.to_datetime(dfnovo['MÊS REFERÊNCIA'], dayfirst=True, errors='coerce')
        
        # Remove linhas vazias ou com datas corrompidas para não quebrar o código
        dfnovo = dfnovo.dropna(subset=['MÊS REFERÊNCIA'])
        
        # Agora sim é 100% seguro usar o .dt
        dfnovo['ANO'] = dfnovo['MÊS REFERÊNCIA'].dt.year
        dfnovo['MÊS'] = dfnovo['MÊS REFERÊNCIA'].dt.month
        
        # REMOVE anomalias futuristas de cara 
        ano_atual = pd.Timestamp.now().year
        dfnovo = dfnovo[dfnovo['ANO'] <= ano_atual]
        
        # PONTO CRÍTICO CORRIGIDO: A verdadeira data máxima da base
        data_maxima_base = dfnovo['MÊS REFERÊNCIA'].max()

    if 'STATUS NO MÊS' in dfnovo.columns:
        dfnovo['STATUS NO MÊS'] = dfnovo['STATUS NO MÊS'].astype(str).str.strip().str.upper()
        
    # 3. Agrupamento de Desligados
    desligamento_setor = dfnovo[dfnovo['STATUS NO MÊS'] == 'DESLIGADO'].groupby(['ANO', 'MÊS', 'SETOR']).size().reset_index(name='QUANTIDADE_DESLIGADOS')
    
    # Só processa se houver desligamentos
    if not desligamento_setor.empty:
        desligamento_setor['DATA'] = pd.to_datetime(desligamento_setor['ANO'].astype(str) + '-' + desligamento_setor['MÊS'].astype(str) + '-01')
        data_minima_base = desligamento_setor['DATA'].min()
    else:
        # Fallback de segurança se a base for vazia de desligados
        return pd.DataFrame(), pd.Series(dtype='category')
    
    # 4. A MAGIA TEMPORAL CORRIGIDA
    # Cria os meses vazios até a data máxima da EMPRESA, e não do último desligado!
    datas_unicas = pd.date_range(start=data_minima_base, end=data_maxima_base, freq='MS')
    setores_unicos = dfnovo['SETOR'].dropna().unique()
    
    idx_completo = pd.MultiIndex.from_product([datas_unicas, setores_unicos], names=['DATA', 'SETOR'])
    df_base = pd.DataFrame(index=idx_completo).reset_index()
    
    df_ml = pd.merge(df_base, desligamento_setor[['DATA', 'SETOR', 'QUANTIDADE_DESLIGADOS']], on=['DATA', 'SETOR'], how='left')
    df_ml['QUANTIDADE_DESLIGADOS'] = df_ml['QUANTIDADE_DESLIGADOS'].fillna(0)
    
    # 5. Lags e Feature Engineering
    df_ml = df_ml.sort_values(by=['SETOR', 'DATA'])
    df_ml['lag_1'] = df_ml.groupby('SETOR')['QUANTIDADE_DESLIGADOS'].shift(1)
    df_ml['lag_2'] = df_ml.groupby('SETOR')['QUANTIDADE_DESLIGADOS'].shift(2)
    df_ml['media_movel_3m'] = df_ml.groupby('SETOR')['lag_1'].transform(lambda x: x.rolling(window=3).mean())
    df_ml['mes_do_ano'] = df_ml['DATA'].dt.month
    
    df_ml = df_ml.dropna()
    df_ml['SETOR_cat'] = df_ml['SETOR'].astype('category').cat.codes
    
    return df_ml, df_ml['SETOR'].astype('category')

def treinar_e_prever(df_ml, cat_setores, meses_para_prever):
    features = ['SETOR_cat', 'lag_1', 'lag_2', 'media_movel_3m', 'mes_do_ano']
    target = 'QUANTIDADE_DESLIGADOS'
    
    modelo = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, objective='count:poisson')
    modelo.fit(df_ml[features], df_ml[target])
    
    mapa_setores = dict(enumerate(cat_setores.cat.categories))
    mapa_reverso = {v: k for k, v in mapa_setores.items()}
    
    df_futuro = df_ml[['DATA', 'SETOR', 'QUANTIDADE_DESLIGADOS']].copy()
    setores_unicos = df_futuro['SETOR'].unique()
    
    for _ in range(meses_para_prever):
        proxima_data = df_futuro['DATA'].max() + pd.DateOffset(months=1)
        novas_linhas = pd.DataFrame({'DATA': proxima_data, 'SETOR': setores_unicos, 'QUANTIDADE_DESLIGADOS': np.nan})
        
        df_futuro = pd.concat([df_futuro, novas_linhas], ignore_index=True)
        df_futuro = df_futuro.sort_values(by=['SETOR', 'DATA']).reset_index(drop=True)
        
        df_futuro['lag_1'] = df_futuro.groupby('SETOR')['QUANTIDADE_DESLIGADOS'].shift(1)
        df_futuro['lag_2'] = df_futuro.groupby('SETOR')['QUANTIDADE_DESLIGADOS'].shift(2)
        df_futuro['media_movel_3m'] = df_futuro.groupby('SETOR')['lag_1'].transform(lambda x: x.rolling(window=3).mean())
        df_futuro['mes_do_ano'] = df_futuro['DATA'].dt.month
        df_futuro['SETOR_cat'] = df_futuro['SETOR'].map(mapa_reverso)
        
        mascara_mes = df_futuro['DATA'] == proxima_data
        X_futuro = df_futuro[mascara_mes][features]
        
        previsoes = np.clip(modelo.predict(X_futuro), 0, None)
        df_futuro.loc[mascara_mes, 'QUANTIDADE_DESLIGADOS'] = previsoes
        
    return df_futuro

# ---------------------------------------------------------
# FLUXO PRINCIPAL DA APLICAÇÃO
# ---------------------------------------------------------
if arquivo_upload is not None:
    # Lendo o arquivo dinamicamente
    if arquivo_upload.name.endswith('.csv'):
        dfnovo = pd.read_csv(arquivo_upload)
    else:
        dfnovo = pd.read_excel(arquivo_upload)
        
    try:
        # Processamento e Modelagem
        with st.spinner("Processando histórico e treinando Inteligência Artificial..."):
            df_ml, cat_setores = processar_dados(dfnovo)
            df_futuro = treinar_e_prever(df_ml, cat_setores, meses_futuro)
            
        ultima_data_historica = df_ml['DATA'].max()
        
        # Tabelas para visualização
        historico_macro = df_ml.groupby('DATA')['QUANTIDADE_DESLIGADOS'].sum().reset_index()
        tabela_futuro = df_futuro[df_futuro['DATA'] > ultima_data_historica].copy()
        futuro_macro = tabela_futuro.groupby('DATA')['QUANTIDADE_DESLIGADOS'].sum().reset_index()
        
        total_previsto = int(futuro_macro['QUANTIDADE_DESLIGADOS'].sum().round())
        
        # ---- MÉTRICAS NO TOPO ----
        col1, col2, col3 = st.columns(3)
        col1.metric("Mês Atual da Base", ultima_data_historica.strftime('%m/%Y'))
        col2.metric(f"Previsão Total ({meses_futuro} meses)", f"~ {total_previsto} estagiários", delta="Alerta de Reposição", delta_color="inverse")
        
        # ---- GRÁFICO PRINCIPAL ----
        st.subheader("📈 Evolução e Tendência de Desligamentos (Empresa)")
        fig, ax = plt.subplots(figsize=(18, 8))
        plt.style.use('seaborn-v0_8-whitegrid')
        
        futuro_plot = pd.concat([historico_macro.iloc[[-1]], futuro_macro])
        
        ax.plot(historico_macro['DATA'], historico_macro['QUANTIDADE_DESLIGADOS'], color='#1f77b4', linewidth=2, marker='o', label='Histórico')
        ax.plot(futuro_plot['DATA'], futuro_plot['QUANTIDADE_DESLIGADOS'], color='#ff7f0e', linewidth=2, linestyle='--', marker='s', label='Previsão')
        ax.fill_between(futuro_plot['DATA'], futuro_plot['QUANTIDADE_DESLIGADOS'], color='#ff7f0e', alpha=0.1)
        
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%Y'))
        plt.xticks(rotation=45)
        plt.ylabel("Total de Desligamentos")
        plt.legend()
        st.pyplot(fig)
        
        # ---- TABELA POR SETOR (visão acumulada) ----
        st.subheader("🎯 Radar de Reposição por Setor")
        
        # MUDANÇA: Agrupamos apenas por setor, somando o risco de todo o período escolhido
        tabela_rh_resumo = tabela_futuro.groupby('SETOR')['QUANTIDADE_DESLIGADOS'].sum().reset_index()
        
        # Como estamos somando vários meses, o piso de 0.5 (meia vaga de probabilidade acumulada) é perfeito
        piso_relevancia = 0.5 
        tabela_filtrada = tabela_rh_resumo[tabela_rh_resumo['QUANTIDADE_DESLIGADOS'] >= piso_relevancia].copy()
        
        # Transforma o risco fracionado em vagas inteiras para repor
        tabela_filtrada['VAGAS_PARA_REPOR'] = np.ceil(tabela_filtrada['QUANTIDADE_DESLIGADOS']).astype(int)
        
        # Ordena do setor com maior necessidade para o menor
        tabela_filtrada = tabela_filtrada.sort_values(by='VAGAS_PARA_REPOR', ascending=False)
        
        # Selecionando colunas finais
        tabela_final_rh = tabela_filtrada[['SETOR', 'VAGAS_PARA_REPOR']]
        
        # Exibição
        if tabela_final_rh.empty:
            st.success("🎉 Ótima notícia! Risco difuso. Nenhum setor tem concentração alta de desligamentos prevista.")
        else:
            st.markdown(f"**Atenção:** Os {tabela_final_rh['VAGAS_PARA_REPOR'].sum()} desligamentos com maior probabilidade estão concentrados nestes setores:")
            st.dataframe(tabela_final_rh, use_container_width=True)
        
            # Botão de Download
            csv = tabela_final_rh.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Baixar Plano de Reposição (CSV)",
                data=csv,
                file_name='radar_reposicao_acumulado.csv',
                mime='text/csv',
            )
            
    except KeyError as e:
        st.error(f"Erro nas colunas do arquivo. Certifique-se de que a base possui as colunas originais (ANO, MÊS, SETOR, STATUS NO MÊS). Erro técnico: {e}")
else:
    st.info("👈 Aguardando o upload do arquivo na barra lateral para gerar o dashboard.")
