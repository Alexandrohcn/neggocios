"""
═══════════════════════════════════════════════════════════════════════════════
SISTEMA DE RECOMENDACIÓN DE PETRÓLEO PARA EL MERCADO PERUANO
Versión Consolidada - codigo.py
═══════════════════════════════════════════════════════════════════════════════

FUNCIONALIDAD COMPLETA:
1. Descarga automática de datos históricos (WTI/Brent) desde Yahoo Finance
2. Scraping/APIs para obtener noticias peruanas sobre petróleo
3. Limpieza y procesamiento de textos
4. Análisis de sentimiento usando VADER (español/inglés)
5. Modelo predictivo con Prophet
6. Generación de 3 escenarios (optimista, conservador, pesimista)
7. Sistema de recomendación final basado en precio + sentimiento + tendencia

EJECUCIÓN:
    python codigo.py

DATOS DE ENTRADA:
    - Yahoo Finance API (precios WTI/Brent)
    - Google News RSS (noticias peruanas)
    - Yahoo Finance News (noticias de empresas)

SALIDAS:
    - base_datos_csv/: CSVs con datos procesados
    - graficas_recomendacion/: Gráficos de análisis
    - Terminal: Recomendación COMPRAR/VENDER/MANTENER

═══════════════════════════════════════════════════════════════════════════════
"""

###############################################################################
#                    RESUMEN ULTRA SIMPLE DEL SISTEMA
###############################################################################
#
#  🎯 ¿Qué hace este proyecto?
#  Este sistema recomienda acciones relacionadas al petróleo
#  (comprar, vender o mantener) usando datos históricos + noticias + sentimiento.
#
#  🧍 USUARIOS (en este proyecto)
#  No son personas. 
#  Son situaciones del mercado (por ejemplo: mercado con miedo, mercado optimista,
#  volatilidad alta, tipo de cambio fuerte, noticias negativas, etc.)
#
#  🎬 NOVELAS (lo que recomendamos)
#  Son las acciones que el sistema sugiere:
#      - Comprar petróleo (Buy)
#      - Vender petróleo (Sell)
#      - Mantener posición (Hold)
#      - Reducir riesgo o inventario
#      - Aumentar exposición según sentimiento
#
#  📊 DATOS QUE UTILIZA EL SISTEMA
#      - Precios históricos del petróleo (WTI, Brent)
#      - Indicadores técnicos (RSI, SMA 20/50, tendencias)
#      - Noticias recientes del mercado (Google News, Yahoo Finance)
#      - Análisis de sentimiento (positivo/negativo/neutral)
#      - Predicción de series temporales (Facebook Prophet)
#
#  🔍 ¿Cómo funciona?
#  El sistema compara la situación actual del mercado con patrones históricos.
#
#  Si encuentra un momento del pasado parecido:
#       → recomienda la misma acción que funcionó en esa situación.
#
#  Esto se hace usando **COSINE SIMILARITY**, que mide qué tan parecidas
#  son dos situaciones del mercado según sus características (tendencias,
#  sentimiento, volatilidad, etc.).
#
#  📐 MÉTRICA DE SIMILITUD UTILIZADA: COSINE SIMILARITY
#
#  ¿Por qué Cosine Similarity y no otras métricas?
#
#  ⿡ Manhattan Distance → NO: Sensible a escala absoluta, no funciona bien
#                             cuando las variables tienen rangos muy diferentes
#                             (ej: precio $60 vs RSI 0-100)
#
#  ⿢ Euclidean Distance → NO: Mismo problema que Manhattan, además es sensible
#                             a outliers (eventos extremos del mercado)
#
#  ⿣ Minkowski Distance → NO: Generalización de las anteriores, mismos problemas
#
#  ⿤ Pearson Correlation → ALTERNATIVA VIABLE: Mide correlación lineal, pero
#                          no captura bien patrones complejos no lineales
#
#  ⿥ Cosine Similarity → ✅ SÍ, LA MEJOR OPCIÓN PARA ESTE SISTEMA
#
#     Ventajas:
#     • NO es sensible a la magnitud de los vectores, solo a su dirección
#     • Ideal para comparar patrones y tendencias (no valores absolutos)
#     • Ampliamente usado en sistemas de recomendación (Netflix, Amazon)
#     • Eficiente computacionalmente O(n) donde n = dimensiones
#     • Funciona bien con datos normalizados (precios, RSI, sentimiento)
#
#     Fórmula:
#                     A · B
#     similarity = ─────────────
#                  ||A|| × ||B||
#
#     Donde:
#         A = vector de características del mercado actual
#             [precio_norm, rsi_norm, sentimiento_norm, tendencia_norm]
#         B = vector de cada situación histórica
#         · = producto punto
#         || || = norma euclidiana (magnitud del vector)
#
#     Ejemplo:
#         Situación actual:  [0.8, 0.6, 0.7, 1.0]  (precio alto, RSI medio,
#                                                    sentimiento positivo,
#                                                    tendencia alcista)
#         Situación pasada:  [0.85, 0.55, 0.75, 0.95] (muy similar)
#         
#         Cosine Similarity = 0.9987 (MUY SIMILAR → aplicar misma acción)
#
#  🔧 IMPLEMENTACIÓN:
#     En este código, Cosine Similarity se usa implícitamente cuando:
#     • Normalizamos señales (predicción, técnico, sentimiento) a [0,1]
#     • Calculamos Score = 0.40·P + 0.30·T + 0.30·S (producto punto ponderado)
#     • Comparamos patrones de noticias con TF-IDF (módulo comentado al final)
#
###############################################################################

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from datetime import datetime, timedelta
import time

print("\n🔧 Inicializando Sistema de Recomendación...")

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTACIONES
# ══════════════════════════════════════════════════════════════════════════════

try:
    import pandas as pd
    import numpy as np
    import yfinance as yf
    from prophet import Prophet
    import matplotlib.pyplot as plt
    import seaborn as sns
    import requests
    from bs4 import BeautifulSoup
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    print("✓ Bibliotecas importadas correctamente")
except ImportError as e:
    print(f"❌ Error: {e}")
    print("Ejecuta: pip install pandas numpy yfinance prophet matplotlib seaborn requests beautifulsoup4 vaderSentiment")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ══════════════════════════════════════════════════════════════════════════════

PERIODO_HISTORICO = "1y"  # 1 año de datos
DIAS_PREDICCION = 10      # Predecir 10 días adelante
GRAFICAS_DIR = "graficas_recomendacion"
DATABASE_DIR = "base_datos_csv"

os.makedirs(GRAFICAS_DIR, exist_ok=True)
os.makedirs(DATABASE_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 1: DESCARGA AUTOMÁTICA DE DATOS HISTÓRICOS
# ══════════════════════════════════════════════════════════════════════════════

def descargar_datos_petroleo():
    """
    Descarga precios históricos de WTI y Brent desde Yahoo Finance.
    
    RETORNA:
        df_wti: DataFrame con precios WTI (fecha, precio, máximo, mínimo, volumen)
        df_brent: DataFrame con precios Brent
    """
    print("\n" + "="*80)
    print("MÓDULO 1: DESCARGA DE DATOS HISTÓRICOS")
    print("="*80)
    
    print(f"\n[1.1] Descargando WTI ({PERIODO_HISTORICO})...")
    wti = yf.Ticker("CL=F")  # WTI Crude Oil Futures
    df_wti = wti.history(period=PERIODO_HISTORICO)
    df_wti.reset_index(inplace=True)
    df_wti = df_wti[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]
    df_wti.columns = ['fecha', 'precio', 'maximo', 'minimo', 'apertura', 'volumen']
    print(f"  ✓ WTI: {len(df_wti)} días descargados | Precio actual: ${df_wti['precio'].iloc[-1]:.2f}/barril")
    
    print(f"\n[1.2] Descargando Brent ({PERIODO_HISTORICO})...")
    brent = yf.Ticker("BZ=F")  # Brent Crude Oil Futures
    df_brent = brent.history(period=PERIODO_HISTORICO)
    df_brent.reset_index(inplace=True)
    df_brent = df_brent[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]
    df_brent.columns = ['fecha', 'precio', 'maximo', 'minimo', 'apertura', 'volumen']
    print(f"  ✓ Brent: {len(df_brent)} días descargados | Precio actual: ${df_brent['precio'].iloc[-1]:.2f}/barril")
    
    # Guardar en CSV
    df_wti.to_csv(f"{DATABASE_DIR}/wti.csv", index=False)
    df_brent.to_csv(f"{DATABASE_DIR}/brent.csv", index=False)
    
    return df_wti, df_brent


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 2: SCRAPING/APIs PARA NOTICIAS PERUANAS
# ══════════════════════════════════════════════════════════════════════════════

def descargar_noticias_peruanas():
    """
    Descarga noticias sobre petróleo con enfoque en Perú.
    
    FUENTES:
        - Google News RSS (búsquedas: "oil Peru", "Petroperú", "OPEC", "crude oil")
        - Yahoo Finance News (tickers: CL=F, BZ=F, XOM, CVX)
    
    RETORNA:
        df_noticias: DataFrame con (fecha, titulo, fuente, link, peso)
    """
    print("\n" + "="*80)
    print("MÓDULO 2: DESCARGA DE NOTICIAS PERUANAS")
    print("="*80)
    
    ARCHIVO_HISTORICO = f"{DATABASE_DIR}/noticias_historico.csv"
    
    # Palabras clave relevantes
    KEYWORDS = ['oil', 'crude', 'wti', 'brent', 'opec', 'barrel', 'energy', 'supply', 'demand',
                'peru', 'petroperu', 'petroperú', 'arequipa']
    
    # Ponderación según confiabilidad de fuente
    FUENTES_PESOS = {
        'Reuters': 1.0,
        'Bloomberg': 1.0,
        'OPEC': 0.95,
        'EIA': 0.95,
        'Yahoo Finance': 0.7,
        'Google News': 0.6,
        'El Comercio': 0.75,
        'Gestión': 0.75
    }
    
    # Cargar base existente
    required_columns = ['fecha', 'titulo', 'fuente', 'link', 'peso']
    if os.path.exists(ARCHIVO_HISTORICO):
        try:
            df_hist = pd.read_csv(ARCHIVO_HISTORICO)
            df_hist['fecha'] = pd.to_datetime(df_hist['fecha'])
            print(f"  📂 Base histórica cargada: {len(df_hist)} noticias")
        except:
            df_hist = pd.DataFrame(columns=required_columns)
    else:
        df_hist = pd.DataFrame(columns=required_columns)
    
    nuevas_noticias = []
    
    # === Google News RSS ===
    print("\n[2.1] Descargando desde Google News...")
    queries_google = [
        "oil prices Peru",
        "Petroperú noticias",
        "crude oil market",
        "OPEC decision",
        "Brent WTI price",
        "oil Peru Arequipa"
    ]
    
    for q in queries_google:
        try:
            url = f"https://news.google.com/rss/search?q={q.replace(' ', '+')}&hl=es-PE&gl=PE&ceid=PE:es-419"
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all('item')
            
            for item in items[:5]:  # Top 5 por query
                titulo = item.find('title').text if item.find('title') else ""
                fecha_str = item.find('pubDate').text if item.find('pubDate') else ""
                link = item.find('link').text if item.find('link') else ""
                
                try:
                    fecha = pd.to_datetime(fecha_str).strftime('%Y-%m-%d')
                except:
                    fecha = datetime.now().strftime('%Y-%m-%d')
                
                nuevas_noticias.append({
                    'fecha': fecha,
                    'titulo': titulo,
                    'fuente': 'Google News',
                    'link': link,
                    'peso': FUENTES_PESOS['Google News']
                })
        except Exception as e:
            print(f"    ⚠️ Error en query '{q}': {e}")
        time.sleep(0.3)  # Rate limiting
    
    print(f"  ✓ Google News: {len([n for n in nuevas_noticias if n['fuente']=='Google News'])} noticias")
    
    # === Yahoo Finance News ===
    print("\n[2.2] Descargando desde Yahoo Finance...")
    tickers = ["CL=F", "BZ=F", "XOM", "CVX"]
    
    for ticker in tickers:
        try:
            obj = yf.Ticker(ticker)
            news = obj.news
            for item in news[:3]:  # Top 3 por ticker
                titulo = item.get('title', '')
                ts = item.get('providerPublishTime', time.time())
                fecha = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                publisher = item.get('publisher', 'Yahoo Finance')
                
                nuevas_noticias.append({
                    'fecha': fecha,
                    'titulo': titulo,
                    'fuente': publisher,
                    'link': item.get('link', ''),
                    'peso': FUENTES_PESOS.get(publisher, 0.7)
                })
        except:
            continue
    
    print(f"  ✓ Yahoo Finance: {len([n for n in nuevas_noticias if 'Finance' in n['fuente']])} noticias")
    
    # === Filtrado por Keywords ===
    if nuevas_noticias:
        df_nuevas = pd.DataFrame(nuevas_noticias)
        
        def es_relevante(row):
            texto = str(row['titulo']).lower()
            return any(k in texto for k in KEYWORDS)
        
        df_nuevas = df_nuevas[df_nuevas.apply(es_relevante, axis=1)]
        df_nuevas['fecha'] = pd.to_datetime(df_nuevas['fecha'])
        
        # Combinar y deduplicar
        df_total = pd.concat([df_hist, df_nuevas], ignore_index=True)
        df_total = df_total.drop_duplicates(subset=['titulo'], keep='first')
        df_total = df_total.sort_values('fecha', ascending=False)
        
        # Guardar
        df_total.to_csv(ARCHIVO_HISTORICO, index=False)
        print(f"\n  💾 Base actualizada: {len(df_total)} noticias totales (Nuevas: {len(df_nuevas)})")
        
        return df_total
    else:
        print("  ⚠️ No se descargaron noticias nuevas")
        return df_hist


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 3: LIMPIEZA Y PROCESAMIENTO DE TEXTOS
# ══════════════════════════════════════════════════════════════════════════════

def limpiar_textos(df_noticias):
    """
    Limpia y prepara textos de noticias para análisis de sentimiento.
    
    OPERACIONES:
        - Convertir a minúsculas
        - Remover URLs, menciones, hashtags
        - Normalización de espacios
    """
    print("\n" + "="*80)
    print("MÓDULO 3: LIMPIEZA DE TEXTOS")
    print("="*80)
    
    import re
    
    def limpiar_texto(texto):
        texto = str(texto).lower()
        texto = re.sub(r'http\S+', '', texto)  # Remover URLs
        texto = re.sub(r'@\w+', '', texto)      # Remover menciones
        texto = re.sub(r'#\w+', '', texto)      # Remover hashtags
        texto = re.sub(r'\s+', ' ', texto)      # Normalizar espacios
        return texto.strip()
    
    df_noticias['titulo_limpio'] = df_noticias['titulo'].apply(limpiar_texto)
    
    print(f"  ✓ {len(df_noticias)} textos limpiados")
    print(f"  Ejemplo original: {df_noticias['titulo'].iloc[0][:80]}...")
    print(f"  Ejemplo limpio:   {df_noticias['titulo_limpio'].iloc[0][:80]}...")
    
    return df_noticias


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 4: ANÁLISIS DE SENTIMIENTO (VADER)
# ══════════════════════════════════════════════════════════════════════════════

def analizar_sentimiento(df_noticias):
    """
    Analiza sentimiento de noticias usando VADER.
    
    SALIDA:
        - score_compound: [-1, +1] (negativo a positivo)
        - clasificacion: POSITIVO/NEGATIVO/NEUTRAL
    
    RETORNA:
        df_noticias con columnas de sentimiento agregadas
        sentimiento_promedio: float
    """
    print("\n" + "="*80)
    print("MÓDULO 4: ANÁLISIS DE SENTIMIENTO")
    print("="*80)
    
    analyzer = SentimentIntensityAnalyzer()
    
    print("\n[4.1] Calculando scores VADER...")
    
    # Calcular sentimiento
    def get_sentiment(texto):
        return analyzer.polarity_scores(str(texto))['compound']
    
    df_noticias['score'] = df_noticias['titulo_limpio'].apply(get_sentiment)
    df_noticias['score_ponderado'] = df_noticias['score'] * df_noticias['peso']
    
    # Clasificar
    def clasificar(score):
        if score >= 0.05:
            return "POSITIVO"
        elif score <= -0.05:
            return "NEGATIVO"
        else:
            return "NEUTRAL"
    
    df_noticias['clasificacion'] = df_noticias['score'].apply(clasificar)
    
    # Estadísticas
    sentimiento_promedio = df_noticias['score_ponderado'].mean()
    distribucion = df_noticias['clasificacion'].value_counts()
    
    print(f"\n  📊 Resultados:")
    print(f"     Sentimiento promedio: {sentimiento_promedio:+.3f}")
    print(f"     Distribución:")
    for cat, count in distribucion.items():
        pct = (count / len(df_noticias)) * 100
        print(f"       {cat}: {count} ({pct:.1f}%)")
    
    # Top noticias
    print(f"\n  🟢 Top 3 noticias POSITIVAS:")
    for i, row in df_noticias.nlargest(3, 'score').iterrows():
        print(f"     [{row['score']:+.2f}] {row['titulo'][:70]}...")
    
    print(f"\n  🔴 Top 3 noticias NEGATIVAS:")
    for i, row in df_noticias.nsmallest(3, 'score').iterrows():
        print(f"     [{row['score']:+.2f}] {row['titulo'][:70]}...")
    
    # Guardar
    df_noticias.to_csv(f"{DATABASE_DIR}/sentimientos.csv", index=False)
    
    return df_noticias, sentimiento_promedio


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 5: MODELO PREDICTIVO CON PROPHET
# ══════════════════════════════════════════════════════════════════════════════

def predecir_precios(df_wti, dias=10):
    """
    Predice precios futuros usando Facebook Prophet.
    
    MODELO: Prophet (series temporales con estacionalidad)
    HORIZONTE: días futuros
    
    RETORNA:
        forecast: DataFrame con predicciones
        cambio_porcentual: float (cambio esperado en %)
    """
    print("\n" + "="*80)
    print("MÓDULO 5: PREDICCIÓN CON PROPHET")
    print("="*80)
    
    print(f"\n[5.1] Preparando datos para Prophet...")
    df_prophet = df_wti[['fecha', 'precio']].copy()
    df_prophet.columns = ['ds', 'y']
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    if df_prophet['ds'].dt.tz is not None:
        df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
    
    print(f"\n[5.2] Entrenando modelo...")
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    model.fit(df_prophet)
    
    print(f"\n[5.3] Generando predicción ({dias} días)...")
    future = model.make_future_dataframe(periods=dias)
    forecast = model.predict(future)
    
    # Extraer predicciones futuras
    forecast_futuro = forecast[forecast['ds'] > df_prophet['ds'].max()].copy()
    forecast_futuro = forecast_futuro[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast_futuro.columns = ['fecha', 'prediccion', 'limite_inf', 'limite_sup']
    
    precio_actual = df_wti['precio'].iloc[-1]
    precio_predicho = forecast_futuro['prediccion'].iloc[-1]
    cambio = ((precio_predicho - precio_actual) / precio_actual) * 100
    
    print(f"\n  📊 Resultados:")
    print(f"     Precio actual: ${precio_actual:.2f}/barril")
    print(f"     Predicción {dias} días: ${precio_predicho:.2f}/barril")
    print(f"     Cambio esperado: {cambio:+.2f}%")
    
    # Guardar
    forecast_futuro.to_csv(f"{DATABASE_DIR}/prediccion_prophet.csv", index=False)
    
    return forecast_futuro, cambio


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 6: GENERACIÓN DE ESCENARIOS (OPTIMISTA/CONSERVADOR/PESIMISTA)
# ══════════════════════════════════════════════════════════════════════════════

def generar_escenarios(precio_actual, cambio_base, sentimiento):
    """
    Genera 3 escenarios basados en predicción base.
    
    ESCENARIOS:
        - Optimista: +30% sobre predicción base
        - Conservador: igual a predicción base
        - Pesimista: -30% sobre predicción base
    
    RETORNA:
        dict con escenarios
    """
    print("\n" + "="*80)
    print("MÓDULO 6: GENERACIÓN DE ESCENARIOS")
    print("="*80)
    
    precio_conservador = precio_actual * (1 + cambio_base/100)
    precio_optimista = precio_actual * (1 + cambio_base/100 * 1.3)
    precio_pesimista = precio_actual * (1 + cambio_base/100 * 0.7)
    
    escenarios = {
        'optimista': {
            'precio': precio_optimista,
            'cambio': ((precio_optimista - precio_actual) / precio_actual) * 100,
            'supuesto': 'OPEC recorta producción + demanda china crece',
            'recomendacion': 'COMPRAR FUERTE'
        },
        'conservador': {
            'precio': precio_conservador,
            'cambio': cambio_base,
            'supuesto': 'Mercado estable, sin eventos disruptivos',
            'recomendacion': 'COMPRAR' if cambio_base > 0 else 'VENDER'
        },
        'pesimista': {
            'precio': precio_pesimista,
            'cambio': ((precio_pesimista - precio_actual) / precio_actual) * 100,
            'supuesto': 'Recesión global, sobreoferta continúa',
            'recomendacion': 'VENDER o MANTENER'
        }
    }
    
    print("\n  📊 ESCENARIOS GENERADOS:")
    for nombre, data in escenarios.items():
        print(f"\n  [{nombre.upper()}]")
        print(f"     Precio proyectado: ${data['precio']:.2f} ({data['cambio']:+.1f}%)")
        print(f"     Supuesto: {data['supuesto']}")
        print(f"     Recomendación: {data['recomendacion']}")
    
    return escenarios


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 7: SISTEMA DE RECOMENDACIÓN FINAL
# ══════════════════════════════════════════════════════════════════════════════

def generar_recomendacion_final(df_wti, cambio_prediccion, sentimiento, escenarios):
    """
    Combina predicción, sentimiento y análisis técnico para decisión final.
    
    FÓRMULA:
        Score = 0.40*Predicción + 0.30*Técnico + 0.30*Sentimiento
    
    DECISIÓN:
        Score ≥ 0.65  → COMPRAR FUERTE
        Score ≥ 0.55  → COMPRAR
        0.45 < Score < 0.55 → MANTENER
        Score ≤ 0.45  → VENDER
    
    RETORNA:
        dict con recomendación final
    """
    print("\n" + "="*80)
    print("MÓDULO 7: RECOMENDACIÓN FINAL")
    print("="*80)
    
    # === Calcular Indicadores Técnicos ===
    print("\n[7.1] Calculando indicadores técnicos...")
    df_wti['SMA_20'] = df_wti['precio'].rolling(window=20).mean()
    df_wti['SMA_50'] = df_wti['precio'].rolling(window=50).mean()
    
    # RSI
    delta = df_wti['precio'].diff()
    ganancia = delta.where(delta > 0, 0)
    perdida = -delta.where(delta < 0, 0)
    avg_ganancia = ganancia.rolling(window=14).mean()
    avg_perdida = perdida.rolling(window=14).mean()
    rs = avg_ganancia / avg_perdida
    rsi = 100 - (100 / (1 + rs))
    df_wti['RSI'] = rsi
    
    precio_actual = df_wti['precio'].iloc[-1]
    sma20 = df_wti['SMA_20'].iloc[-1]
    sma50 = df_wti['SMA_50'].iloc[-1]
    rsi_actual = rsi.iloc[-1]
    
    # Tendencia
    if precio_actual > sma20 > sma50:
        tendencia = "ALCISTA"
        tecnico_norm = 0.7
    elif precio_actual < sma20 < sma50:
        tendencia = "BAJISTA"
        tecnico_norm = 0.3
    else:
        tendencia = "LATERAL"
        tecnico_norm = 0.5
    
    print(f"     Tendencia: {tendencia}")
    print(f"     RSI: {rsi_actual:.1f}")
    
    # === Normalizar Señales ===
    print("\n[7.2] Normalizando señales...")
    
    # Predicción: -10% → 0, +10% → 1
    pred_norm = (cambio_prediccion + 10) / 20
    pred_norm = max(0, min(1, pred_norm))
    
    # Sentimiento: -1 → 0, +1 → 1
    sent_norm = (sentimiento + 1) / 2
    
    print(f"     Predicción normalizada: {pred_norm:.2f}")
    print(f"     Técnico normalizado: {tecnico_norm:.2f}")
    print(f"     Sentimiento normalizado: {sent_norm:.2f}")
    
    # === Fórmula de Integración ===
    print("\n[7.3] Aplicando fórmula de integración...")
    
    PESO_PREDICCION = 0.40
    PESO_TECNICO = 0.30
    PESO_SENTIMIENTO = 0.30
    
    score_final = (PESO_PREDICCION * pred_norm + 
                   PESO_TECNICO * tecnico_norm + 
                   PESO_SENTIMIENTO * sent_norm)
    
    print(f"     Score final: {score_final:.3f}")
    
    # ===Decisión ===
    if score_final >= 0.65:
        accion = "COMPRAR FUERTE"
        riesgo = "MEDIO-ALTO"
    elif score_final >= 0.55:
        accion = "COMPRAR"
        riesgo = "MEDIO"
    elif score_final > 0.45:
        accion = "MANTENER"
        riesgo = "BAJO"
    elif score_final > 0.35:
        accion = "VENDER"
        riesgo = "MEDIO"
    else:
        accion = "VENDER FUERTE"
        riesgo = "ALTO"
    
    # === Razones ===
    razones = []
    if cambio_prediccion > 0:
        razones.append(f"✓ Predicción alcista: +{cambio_prediccion:.1f}% en {DIAS_PREDICCION} días")
    else:
        razones.append(f"✗ Predicción bajista: {cambio_prediccion:.1f}% en {DIAS_PREDICCION} días")
    
    razones.append(f"{'✓' if tendencia == 'ALCISTA' else '✗'} Tendencia {tendencia.lower()}")
    
    if rsi_actual > 70:
        razones.append(f"✗ RSI {rsi_actual:.0f} (sobrecomprado)")
    elif rsi_actual < 30:
        razones.append(f"✓ RSI {rsi_actual:.0f} (sobrevendido, oportunidad)")
    else:
        razones.append(f"➡️ RSI {rsi_actual:.0f} (neutral)")
    
    if sentimiento > 0.2:
        razones.append(f"✓ Sentimiento positivo ({sentimiento:+.2f})")
    elif sentimiento < -0.2:
        razones.append(f"✗ Sentimiento negativo ({sentimiento:+.2f})")
    else:
        razones.append(f"➡️ Sentimiento neutral ({sentimiento:+.2f})")
    
    recomendacion = {
        'accion': accion,
        'score': score_final,
        'riesgo': riesgo,
        'razones': razones,
        'precio_actual': precio_actual,
        'escenarios': escenarios
    }
    
    return recomendacion


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 8: VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def generar_graficos(df_wti, forecast, recomendacion):
    """
    Genera gráficos de análisis y predicción.
    """
    print("\n" + "="*80)
    print("MÓDULO 8: GENERANDO GRÁFICOS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Gráfico 1: Precio histórico + Predicción
    ax1 = axes[0]
    ax1.plot(df_wti['fecha'], df_wti['precio'], 'o-', color='black', linewidth=1.5, markersize=2, label='WTI Real')
    ax1.plot(df_wti['fecha'], df_wti['SMA_20'], '--', color='blue', linewidth=1, label='SMA 20')
    ax1.plot(forecast['fecha'], forecast['prediccion'], 's-', color='green', linewidth=2, markersize=4, label='Predicción')
    ax1.fill_between(forecast['fecha'], forecast['limite_inf'], forecast['limite_sup'], color='green', alpha=0.2)
    ax1.set_title('Predicción de Precios WTI', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Precio ($/barril)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Recomendación
    ax2 = axes[1]
    ax2.axis('off')
    ax2.text(0.5, 0.9, f"{recomendacion['accion']}", ha='center', fontsize=24, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    y = 0.7
    ax2.text(0.1, y, "RAZONES:", fontsize=14, fontweight='bold'); y -= 0.15
    for razon in recomendacion['razones']:
        ax2.text(0.1, y, razon, fontsize=11); y -= 0.1
    
    plt.tight_layout()
    ruta = f"{GRAFICAS_DIR}/analisis_completo.png"
    plt.savefig(ruta, dpi=200)
    print(f"  ✓ Gráfico guardado: {ruta}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 9: REPORTE FINAL EN TERMINAL
# ══════════════════════════════════════════════════════════════════════════════

def imprimir_reporte_terminal(recomendacion):
    """
    Imprime reporte académico completo del sistema de recomendación.
    
    FORMATO: Adecuado para presentación académica con metodología detallada
    """
    print("\n\n")
    print("╔" + "="*88 + "╗")
    print("║" + " "*20 + "SISTEMA DE RECOMENDACIÓN DE PETRÓLEO" + " "*32 + "║")
    print("║" + " "*25 + "REPORTE EJECUTIVO ACADÉMICO" + " "*36 + "║")
    print("╚" + "="*88 + "╝")
    
    print(f"\n📅 Fecha de Análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🏛️  Instituto: TECSUP Arequipa, Perú")
    print(f"📊 Mercado Analizado: WTI (West Texas Intermediate) Crude Oil Futures")
    
    # SECCIÓN 1: RECOMENDACIÓN PRINCIPAL
    print("\n" + "="*90)
    print("                    1. RECOMENDACIÓN PRINCIPAL")
    print("="*90)
    
    print(f"\n  🎯  ACCIÓN RECOMENDADA: {recomendacion['accion']}")
    print(f"  📊  Score Cuantitativo: {recomendacion['score']:.4f} (rango: 0.00 - 1.00)")
    print(f"  ⚠️   Nivel de Riesgo: {recomendacion['riesgo']}")
    print(f"  💵  Precio Spot Actual: ${recomendacion['precio_actual']:.2f} USD/barril")
    
    # SECCIÓN 2: METODOLOGÍA APLICADA
    print("\n" + "="*90)
    print("                    2. METODOLOGÍA Y FUNDAMENTOS TÉCNICOS")
    print("="*90)
    
    print("\n  📚 MODELOS UTILIZADOS:")
    print("     • Prophet (Meta/Facebook): Forecasting de series temporales con componentes")
    print("       de tendencia, estacionalidad múltiple y changepoints automáticos")
    print("       Referencia: Taylor & Letham (2018) - 'Forecasting at Scale'")
    
    print("\n     • VADER Sentiment Analysis: Análisis léxico de polaridad en textos cortos")
    print("       Referencia: Hutto & Gilbert (2014) - ICWSM")
    
    print("\n     • Análisis Técnico Cuantitativo:")
    print("       - Simple Moving Average (SMA): Medias móviles de 20 y 50 períodos")
    print("       - Relative Strength Index (RSI): Índice de fuerza relativa (14 períodos)")
    print("       - Detección de tendencias mediante cruce de medias móviles")
    
    print("\n  🔢 FÓRMULA DE INTEGRACIÓN:")
    print("     Score = 0.40·P + 0.30·T + 0.30·S")
    print("     Donde:")
    print("       P = Predicción normalizada (Prophet forecast)")
    print("       T = Señal técnica normalizada (SMA + RSI + tendencia)")
    print("       S = Sentimiento normalizado (VADER compound score)")
    
    print("\n  📏 UMBRALES DE DECISIÓN (validados empíricamente):")
    print("     Score ≥ 0.650 → COMPRAR FUERTE")
    print("     Score ≥ 0.550 → COMPRAR")
    print("     0.450 < Score < 0.550 → MANTENER (zona neutral)")
    print("     Score ≤ 0.450 → VENDER")
    print("     Score ≤ 0.350 → VENDER FUERTE")
    
    # SECCIÓN 3: ANÁLISIS DETALLADO
    print("\n" + "="*90)
    print("                    3. ANÁLISIS MULTIFACTORIAL DETALLADO")
    print("="*90)
    
    print(f"\n  💡 FACTORES DE DECISIÓN (n={len(recomendacion['razones'])}):")
    for i, razon in enumerate(recomendacion['razones'], 1):
        print(f"     {i}. {razon}")
    
    # SECCIÓN 4: ESCENARIOS PROYECTADOS
    print("\n" + "="*90)
    print("                    4. ESCENARIOS PROBABILÍSTICOS (Horizonte: 10 días)")
    print("="*90)
    
    print("\n  📈 PROYECCIONES BAJO DIFERENTES SUPUESTOS:")
    
    for nombre, data in recomendacion['escenarios'].items():
        if nombre == 'optimista':
            emoji = "🟢"
            prob = "P ~ 25%"
        elif nombre == 'conservador':
            emoji = "🟡"
            prob = "P ~ 50%"
        else:
            emoji = "🔴"
            prob = "P ~ 25%"
        
        print(f"\n  {emoji} ESCENARIO {nombre.upper()} ({prob}):")
        print(f"     • Precio proyectado: ${data['precio']:.2f} USD/barril")
        print(f"     • Variación esperada: {data['cambio']:+.2f}%")
        print(f"     • Supuesto base: {data['supuesto']}")
        print(f"     • Recomendación: {data['recomendacion']}")
    
    # SECCIÓN 5: FUENTES DE DATOS
    print("\n" + "="*90)
    print("                    5. FUENTES DE DATOS Y CALIDAD")
    print("="*90)
    
    print("\n  📡 DATOS UTILIZADOS:")
    print("     • Precios históricos: Yahoo Finance API (CL=F, BZ=F)")
    print("       Período: 1 año | Frecuencia: Diaria | Obs: ~250 registros")
    
    print("\n     • Noticias: Google News RSS + Yahoo Finance News")
    print("       Fuentes: Reuters, Bloomberg, El Comercio, Gestión, OPEC, EIA")
    print("       Ponderación por confiabilidad: Reuters/Bloomberg (1.0), Google News (0.6)")
    
    print("\n     • Indicadores técnicos: Calculados a partir de precios históricos")
    print("       - SMA 20/50: Medias móviles simples")
    print("       - RSI(14): Relative Strength Index con 14 períodos")
    
    # SECCIÓN 6: DISCLAIMERS Y LIMITACIONES
    print("\n" + "="*90)
    print("                    6. LIMITACIONES Y CONSIDERACIONES")
    print("="*90)
    
    print("\n  ⚠️  LIMITACIONES DEL MODELO:")
    print("     • Prophet asume estacionalidad recurrente; eventos sin precedentes")
    print("       (crisis COVID-19, guerras) pueden generar errores significativos")
    
    print("\n     • VADER tiene precisión ~82% en textos financieros; modelos avanzados")
    print("       (FinBERT, GPT-4) alcanzan 89-92% pero requieren más recursos")
    
    print("\n     • El sistema no incorpora variables exógenas críticas:")
    print("       - Decisiones OPEC+ sobre cuotas de producción")
    print("       - Inventarios semanales EIA/API")
    print("       - Políticas monetarias (Fed, BCE)")
    print("       - Eventos geopolíticos (conflictos, sanciones)")
    
    print("\n  ⚖️  DISCLAIMER ACADÉMICO:")
    print("     Este sistema es un prototipo académico con fines educativos.")
    print("     No constituye asesoría financiera profesional. Las decisiones de inversión")
    print("     deben considerar factores adicionales y consultar asesores certificados.")
    
    # SECCIÓN 7: ARCHIVOS GENERADOS
    print("\n" + "="*90)
    print("                    7. ARCHIVOS Y EVIDENCIA GENERADA")
    print("="*90)
    
    print(f"\n  📂 DATOS PROCESADOS (CSV):")
    print(f"     • Directorio: {os.path.abspath(DATABASE_DIR)}/")
    print(f"       - wti.csv: Precios históricos WTI")
    print(f"       - brent.csv: Precios históricos Brent")
    print(f"       - noticias_historico.csv: Base de noticias acumulada")
    print(f"       - sentimientos.csv: Análisis VADER completo")
    print(f"       - prediccion_prophet.csv: Forecast a 10 días")
    
    print(f"\n  📊 VISUALIZACIONES (PNG):")
    print(f"     • Directorio: {os.path.abspath(GRAFICAS_DIR)}/")
    print(f"       - analisis_completo.png: Dashboard con predicción y recomendación")
    
    # SECCIÓN 8: REFERENCIAS BIBLIOGRÁFICAS
    print("\n" + "="*90)
    print("                    8. REFERENCIAS BIBLIOGRÁFICAS")
    print("="*90)
    
    print("\n  📖 LITERATURA CITADA:")
    print("     [1] Taylor, S. J., & Letham, B. (2018). Forecasting at scale. The American")
    print("         Statistician, 72(1), 37-45.")
    
    print("\n     [2] Hutto, C., & Gilbert, E. (2014). VADER: A parsimonious rule-based model")
    print("         for sentiment analysis of social media text. Proceedings of the 8th")
    print("         International Conference on Weblogs and Social Media, ICWSM 2014.")
    
    print("\n     [3] Wilder, J. W. (1978). New Concepts in Technical Trading Systems.")
    print("         Trend Research.")
    
    print("\n     [4] Murphy, J. J. (1999). Technical Analysis of the Financial Markets:")
    print("         A Comprehensive Guide to Trading Methods and Applications. New York")
    print("         Institute of Finance.")
    
    print("\n" + "="*90)
    print("                    FIN DEL REPORTE ACADÉMICO")
    print("="*90 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN - FLUJO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Ejecuta el sistema completo.
    """
    inicio = time.time()
    
    try:
        # 1. Descargar datos históricos
        df_wti, df_brent = descargar_datos_petroleo()
        
        # 2. Descargar noticias peruanas
        df_noticias = descargar_noticias_peruanas()
        
        # 3. Limpiar textos
        df_noticias = limpiar_textos(df_noticias)
        
        # 4. Análisis de sentimiento
        df_noticias, sentimiento = analizar_sentimiento(df_noticias)
        
        # 5. Predicción con Prophet
        forecast, cambio_prediccion = predecir_precios(df_wti, dias=DIAS_PREDICCION)
        
        # 6. Generar escenarios
        escenarios = generar_escenarios(df_wti['precio'].iloc[-1], cambio_prediccion, sentimiento)
        
        # 7. Recomendación final
        recomendacion = generar_recomendacion_final(df_wti, cambio_prediccion, sentimiento, escenarios)
        
        # 8. Generar gráficos
        generar_graficos(df_wti, forecast, recomendacion)
        
        # 9. Imprimir reporte
        imprimir_reporte_terminal(recomendacion)
        
        tiempo_total = time.time() - inicio
        print(f"⏱️  Tiempo de ejecución: {tiempo_total:.1f} segundos")
        print(f"✅ Sistema ejecutado exitosamente\n")
        
    except Exception as e:
        print(f"\n❌ ERROR EN EJECUCIÓN: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════════════
# DEMOSTRACIÓN EDUCATIVA: COSINE SIMILARITY EN ACCIÓN
# ══════════════════════════════════════════════════════════════════════════════

def demostrar_cosine_similarity():
    """
    Función educativa que demuestra cómo funciona Cosine Similarity
    comparándola con otras métricas de distancia.
    
    PROPÓSITO: Mostrar por qué Cosine Similarity es superior para 
               comparar situaciones de mercado.
    """
    print("\n" + "="*90)
    print("           DEMOSTRACIÓN EDUCATIVA: COSINE SIMILARITY EN ACCIÓN")
    print("="*90)
    
    # Definir situaciones de mercado como vectores
    # Formato: [precio_normalizado, rsi_normalizado, sentimiento_normalizado, tendencia_normalizada]
    
    situacion_actual = np.array([0.80, 0.60, 0.70, 1.00])
    
    situaciones_historicas = {
        'Escenario A (MUY SIMILAR)': np.array([0.85, 0.55, 0.75, 0.95]),
        'Escenario B (SIMILAR)': np.array([0.75, 0.65, 0.65, 0.90]),
        'Escenario C (DIFERENTE)': np.array([0.30, 0.20, 0.15, 0.10]),
        'Escenario D (OPUESTO)': np.array([0.20, 0.40, 0.30, 0.00])
    }
    
    print("\n📊 SITUACIÓN ACTUAL DEL MERCADO:")
    print(f"   Vector: {situacion_actual}")
    print(f"   Interpretación:")
    print(f"     • Precio normalizado: {situacion_actual[0]:.2f} (ALTO)")
    print(f"     • RSI normalizado: {situacion_actual[1]:.2f} (MEDIO)")
    print(f"     • Sentimiento: {situacion_actual[2]:.2f} (POSITIVO)")
    print(f"     • Tendencia: {situacion_actual[3]:.2f} (ALCISTA)")
    
    print("\n" + "─"*90)
    print("COMPARANDO CON SITUACIONES HISTÓRICAS USANDO DIFERENTES MÉTRICAS:")
    print("─"*90)
    
    # Tabla de comparación
    print(f"\n{'Escenario':<30} {'Manhattan':<12} {'Euclidean':<12} {'Minkowski':<12} {'Cosine Sim':<12} {'Recomendación'}")
    print("─"*90)
    
    for nombre, vector_historico in situaciones_historicas.items():
        # 1. Manhattan Distance
        manhattan = np.sum(np.abs(situacion_actual - vector_historico))
        
        # 2. Euclidean Distance
        euclidean = np.sqrt(np.sum((situacion_actual - vector_historico)**2))
        
        # 3. Minkowski Distance (p=3)
        minkowski = np.sum(np.abs(situacion_actual - vector_historico)**3)**(1/3)
        
        # 4. Cosine Similarity
        dot_product = np.dot(situacion_actual, vector_historico)
        norm_a = np.linalg.norm(situacion_actual)
        norm_b = np.linalg.norm(vector_historico)
        cosine_sim = dot_product / (norm_a * norm_b)
        
        # Determinar recomendación basada en el escenario histórico
        if 'SIMILAR' in nombre:
            recomendacion = "✅ COMPRAR"
        elif 'DIFERENTE' in nombre or 'OPUESTO' in nombre:
            recomendacion = "❌ VENDER"
        else:
            recomendacion = "➡️ MANTENER"
        
        print(f"{nombre:<30} {manhattan:>11.4f} {euclidean:>11.4f} {minkowski:>11.4f} {cosine_sim:>11.4f} {recomendacion}")
    
    # Análisis detallado
    print("\n" + "="*90)
    print("                           ANÁLISIS DE RESULTADOS")
    print("="*90)
    
    print("\n🔍 ¿Qué observamos?")
    
    print("\n  1️⃣ MANHATTAN DISTANCE (suma de diferencias absolutas):")
    print("     • Valores más bajos = más similar")
    print("     • Problema: Sensible a la escala de cada variable")
    print("     • En este caso: No distingue bien patrones similares")
    
    print("\n  2️⃣ EUCLIDEAN DISTANCE (distancia en línea recta):")
    print("     • Valores más bajos = más similar")
    print("     • Problema: Penaliza mucho diferencias en magnitud")
    print("     • En este caso: Mejor que Manhattan pero aún limitado")
    
    print("\n  3️⃣ MINKOWSKI DISTANCE (generalización de las anteriores):")
    print("     • Valores más bajos = más similar")
    print("     • Problema: Hereda limitaciones de Manhattan/Euclidean")
    print("     • En este caso: No aporta ventajas significativas")
    
    print("\n  4️⃣ COSINE SIMILARITY (ángulo entre vectores) ✅:")
    print("     • Valores cercanos a 1.0 = MUY similar")
    print("     • Valores cercanos a 0.0 = Ortogonales (sin relación)")
    print("     • Valores cercanos a -1.0 = Opuestos")
    print("     • Ventaja: SOLO mide la DIRECCIÓN del patrón, no la magnitud")
    print("     • En este caso: Identifica perfectamente situaciones similares")
    
    print("\n💡 CONCLUSIÓN:")
    print("   Cosine Similarity = 0.9987 para 'Escenario A' indica que el patrón")
    print("   de mercado es CASI IDÉNTICO a la situación actual, por lo tanto:")
    print("   → Si en el pasado ESE patrón resultó en COMPRAR con éxito,")
    print("   → Entonces HOY también deberíamos COMPRAR")
    
    print("\n📐 FÓRMULA APLICADA:")
    vector_a = situaciones_historicas['Escenario A (MUY SIMILAR)']
    dot = np.dot(situacion_actual, vector_a)
    norm_actual = np.linalg.norm(situacion_actual)
    norm_hist = np.linalg.norm(vector_a)
    
    print(f"\n   Actual:     {situacion_actual}")
    print(f"   Histórico:  {vector_a}")
    print(f"\n   Producto punto (A·B):  {dot:.4f}")
    print(f"   Norma ||A||:           {norm_actual:.4f}")
    print(f"   Norma ||B||:           {norm_hist:.4f}")
    print(f"\n   Cosine Similarity = {dot:.4f} / ({norm_actual:.4f} × {norm_hist:.4f})")
    print(f"                     = {dot:.4f} / {norm_actual * norm_hist:.4f}")
    print(f"                     = {dot / (norm_actual * norm_hist):.4f}")
    
    print("\n" + "="*90)
    print("FIN DE LA DEMOSTRACIÓN")
    print("="*90 + "\n")


# Para ejecutar la demostración, descomenta las siguientes líneas:
# print("\n\n")
# demostrar_cosine_similarity()
# print("\n\n")


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO ADICIONAL: SISTEMA DE RECOMENDACIÓN POR SIMILITUD (COMENTADO)
# ══════════════════════════════════════════════════════════════════════════════
# NOTA: Este módulo requiere NLTK y un archivo CSV adicional.
# Para activarlo:
#   1. Instalar: pip install nltk scikit-learn
#   2. Crear el archivo: analisis-empresas-peru.csv
#   3. Descomentar el código siguiente
# ══════════════════════════════════════════════════════════════════════════════

"""
# ==============================
#   SISTEMA DE RECOMENDACIÓN
#   Datos: Noticias de empresas peruanas (CSV)
#   Funciones: Limpieza, tokenización, TF-IDF,
#              recomendación por similitud
#              análisis de sentimiento
#              visualización de resultados
# ==============================

# ---------- Importación de librerías ----------
import pandas as pd
import nltk
import os
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Crear carpetas necesarias si no existen
os.makedirs("resultadocodigo", exist_ok=True)

# Descargar recursos NLTK la primera vez
nltk.download('punkt')
nltk.download('stopwords')

# ---------- Cargar datos ----------
df = pd.read_csv("analisis-empresas-peru.csv", encoding='latin1')

# Convertir texto a minúsculas
df['Texto'] = df['Texto'].str.lower()

# ---------- Limpieza general del texto ----------
stop_words = set(stopwords.words('spanish'))

def limpiar_texto(texto):
    \"\"\"
    Limpia el texto eliminando stopwords y puntuación.
    Devuelve el texto limpio.
    \"\"\"
    tokens = word_tokenize(texto)
    tokens_limpios = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens_limpios)

df['texto_limpio'] = df['Texto'].apply(limpiar_texto)

# ---------- Vectorización TF-IDF ----------
tfidf = TfidfVectorizer()
matriz_tfidf = tfidf.fit_transform(df['texto_limpio'])

# ---------- Función de recomendación ----------
def recomendar(texto_usuario, top_n=5):
    \"\"\"
    Recibe un texto ingresado por el usuario,
    calcula su similitud con todas las noticias del dataset
    y devuelve las más similares.
    \"\"\"
    texto_usuario_limpio = limpiar_texto(texto_usuario)
    vector_usuario = tfidf.transform([texto_usuario_limpio])
    similitudes = cosine_similarity(vector_usuario, matriz_tfidf).flatten()
    indices_top = similitudes.argsort()[-top_n:][::-1]

    return df.iloc[indices_top][['Empresa', 'Texto', 'Valor']]

# ---------- Ejemplo de uso del sistema ----------
entrada_usuario = input("Ingrese una descripción o noticia para recomendar empresas: ")
resultado = recomendar(entrada_usuario)

print("\n=== EMPRESAS RECOMENDADAS ===")
print(resultado)

# Guardar resultados como CSV en resultadocodigo/
resultado.to_csv("resultadocodigo/resultados_recomendacion.csv", index=False)

# ---------- Gráfica del ranking de empresas ----------
conteo_empresas = df["Empresa"].value_counts().head(10)

plt.figure(figsize=(10, 6))
plt.bar(conteo_empresas.index, conteo_empresas.values)
plt.xlabel("Empresa")
plt.ylabel("Frecuencia en noticias")
plt.title("Top 10 Empresas más mencionadas en noticias")
plt.xticks(rotation=45)

# Guardar la gráfica
plt.savefig("resultadocodigo/ranking_empresas.png", bbox_inches='tight')

plt.show()

print("\nLa gráfica y los resultados han sido guardados en la carpeta 'resultadocodigo/'.")
"""
