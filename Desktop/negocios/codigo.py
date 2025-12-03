"""
═══════════════════════════════════════════════════════════════════════════════
           SISTEMA DE RECOMENDACIÓN DE PETRÓLEO
           Basado en Usuarios (Situaciones) y Novelas (Acciones)
═══════════════════════════════════════════════════════════════════════════════

🎯 CONCEPTO CENTRAL:
   - USUARIOS = Situaciones del mercado petrolero
   - NOVELAS = Acciones recomendadas (Buy, Sell, Hold)
   - MÉTRICA = Cosine Similarity (compara patrones, no magnitudes)

📊 FLUJO DEL SISTEMA:
   1. Definir situaciones históricas (usuarios) y sus acciones exitosas (novelas)
   2. Caracterizar la situación actual del mercado
   3. Comparar con situaciones históricas usando Cosine Similarity
   4. Recomendar la acción (novela) de la situación más similar

═══════════════════════════════════════════════════════════════════════════════
"""

import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yfinance as yf
from sklearn.metrics.pairwise import cosine_similarity

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN Y DIRECTORIOS
# ══════════════════════════════════════════════════════════════════════════════

DIRECTORIO_RESULTADOS = "resultadocodigo"
os.makedirs(DIRECTORIO_RESULTADOS, exist_ok=True)

print("\n🔧 Inicializando Sistema de Recomendación de Petróleo...")
print(f"📂 Resultados se guardarán en: {DIRECTORIO_RESULTADOS}/\n")

# ══════════════════════════════════════════════════════════════════════════════
# PARTE 1: DEFINICIÓN DE "USUARIOS" (SITUACIONES DEL MERCADO)
# ══════════════════════════════════════════════════════════════════════════════

print("="*90)
print("PARTE 1: DEFINICIÓN DE USUARIOS (SITUACIONES DEL MERCADO)")
print("="*90)

"""
En un sistema de recomendación tradicional:
   - Usuarios = personas que ven películas
   - Películas = lo que se recomienda

En NUESTRO sistema:
   - Usuarios = SITUACIONES del mercado petrolero
   - Novelas = ACCIONES que debemos tomar

Cada "usuario" (situación) se representa como un VECTOR de características:
   [precio_tendencia, volatilidad, sentimiento, demanda, inventarios, riesgo_geopolitico]
"""

# Base de datos histórica: USUARIOS (situaciones pasadas del mercado)
# Cada fila es una situación histórica con su vector de características

usuarios_historicos = {
    # Formato: [precio↗, volatilidad, sentimiento, demanda, inventarios, riesgo_geo]
    # Valores normalizados entre 0 (bajo) y 1 (alto)
    
    'USUARIO_01_MercadoConMiedo': {
        'vector': np.array([0.30, 0.85, 0.15, 0.40, 0.95, 0.80]),
        'descripcion': 'Mercado con miedo por crisis geopolítica',
        'contexto': 'Guerra en Medio Oriente, inventarios altos, demanda baja',
        'fecha': '2023-10-15'
    },
    
    'USUARIO_02_MercadoOptimista': {
        'vector': np.array([0.75, 0.30, 0.85, 0.80, 0.35, 0.20]),
        'descripcion': 'Mercado optimista con fuerte demanda',
        'contexto': 'Recuperación económica China, inventarios bajos',
        'fecha': '2023-03-20'
    },
    
    'USUARIO_03_VolatilidadAlta': {
        'vector': np.array([0.50, 0.95, 0.40, 0.60, 0.50, 0.75]),
        'descripcion': 'Alta volatilidad por incertidumbre OPEP',
        'contexto': 'Decisión de producción OPEP+ inminente',
        'fecha': '2023-06-01'
    },
    
    'USUARIO_04_RecorteOPEP': {
        'vector': np.array([0.85, 0.55, 0.90, 0.75, 0.25, 0.30]),
        'descripcion': 'OPEP+ anuncia recorte de producción',
        'contexto': 'Recorte de 2M barriles/día, mercado alcista',
        'fecha': '2023-04-05'
    },
    
    'USUARIO_05_CrisisRecesion': {
        'vector': np.array([0.20, 0.70, 0.10, 0.25, 0.90, 0.60]),
        'descripcion': 'Temor a recesión global reduce demanda',
        'contexto': 'Fed sube tasas, pronósticos negativos de crecimiento',
        'fecha': '2023-07-12'
    },
    
    'USUARIO_06_AltaDemandaVerano': {
        'vector': np.array([0.80, 0.40, 0.70, 0.90, 0.30, 0.25]),
        'descripcion': 'Temporada alta de demanda (verano USA)',
        'contexto': 'Driving season, inventarios en mínimos estacionales',
        'fecha': '2023-06-15'
    },
    
    'USUARIO_07_ColapsoPrecio': {
        'vector': np.array([0.10, 0.90, 0.05, 0.20, 0.95, 0.85]),
        'descripcion': 'Colapso de precio por sobreoferta',
        'contexto': 'Shale oil USA en máximos históricos, demanda débil',
        'fecha': '2023-11-08'
    }
}

print(f"\n✓ Definidos {len(usuarios_historicos)} USUARIOS (situaciones históricas del mercado)\n")

# Mostrar algunos ejemplos
for i, (usuario_id, datos) in enumerate(list(usuarios_historicos.items())[:3], 1):
    print(f"  {i}. {usuario_id}")
    print(f"     Descripción: {datos['descripcion']}")
    print(f"     Vector: {datos['vector']}")
    print(f"     Contexto: {datos['contexto']}\n")

# ══════════════════════════════════════════════════════════════════════════════
# PARTE 2: DEFINICIÓN DE "NOVELAS" (ACCIONES RECOMENDADAS)
# ══════════════════════════════════════════════════════════════════════════════

print("="*90)
print("PARTE 2: DEFINICIÓN DE NOVELAS (ACCIONES RECOMENDADAS)")
print("="*90)

"""
Cada "novela" es una ACCIÓN que el sistema puede recomendar.

En sistemas tradicionales:
   - Una película tiene título, género, duración
   
En NUESTRO sistema:
   - Una "novela" es una acción: COMPRAR, VENDER, MANTENER, etc.
"""

# Base de datos: NOVELAS (acciones posibles)
novelas_disponibles = {
    'COMPRAR_FUERTE': {
        'accion': 'COMPRAR PETRÓLEO',
        'nivel': 'AGRESIVO',
        'explicacion': 'Comprar contratos de futuros, aumentar exposición',
        'riesgo': 'MEDIO-ALTO',
        'horizonte': '3-6 meses'
    },
    
    'COMPRAR_MODERADO': {
        'accion': 'COMPRAR PETRÓLEO',
        'nivel': 'MODERADO',
        'explicacion': 'Comprar gradualmente, aprovechar caídas',
        'riesgo': 'MEDIO',
        'horizonte': '1-3 meses'
    },
    
    'MANTENER': {
        'accion': 'MANTENER POSICIÓN',
        'nivel': 'NEUTRAL',
        'explicacion': 'No tomar acción, esperar señales más claras',
        'riesgo': 'BAJO',
        'horizonte': '2-4 semanas'
    },
    
    'VENDER_MODERADO': {
        'accion': 'VENDER PETRÓLEO',
        'nivel': 'MODERADO',
        'explicacion': 'Reducir posiciones gradualmente, tomar ganancias',
        'riesgo': 'MEDIO',
        'horizonte': '1-2 meses'
    },
    
    'VENDER_FUERTE': {
        'accion': 'VENDER PETRÓLEO',
        'nivel': 'AGRESIVO',
        'explicacion': 'Cerrar posiciones rápidamente, proteger capital',
        'riesgo': 'ALTO',
        'horizonte': '2-3 semanas'
    },
    
    'COBERTURA': {
        'accion': 'CUBRIR RIESGO',
        'nivel': 'DEFENSIVO',
        'explicacion': 'Hedging con opciones, proteger cartera',
        'riesgo': 'BAJO',
        'horizonte': '1-6 meses'
    }
}

print(f"\n✓ Definidas {len(novelas_disponibles)} NOVELAS (acciones posibles)\n")

for i, (novela_id, datos) in enumerate(novelas_disponibles.items(), 1):
    print(f"  {i}. {novela_id}")
    print(f"     Acción: {datos['accion']} ({datos['nivel']})")
    print(f"     Explicación: {datos['explicacion']}\n")

# ══════════════════════════════════════════════════════════════════════════════
# PARTE 3: MAPEO USUARIOS → NOVELAS (HISTORIAL DE ÉXITOS)
# ══════════════════════════════════════════════════════════════════════════════

print("="*90)
print("PARTE 3: MAPEO HISTÓRICO (Qué acción funcionó en cada situación)")
print("="*90)

"""
Este es el CONOCIMIENTO del sistema:
   - Para cada USUARIO (situación pasada)
   - Sabemos qué NOVEL (acción) fue exitosa

El sistema USA ESTE HISTORIAL para recomendar en situaciones nuevas.
"""

# Historial: qué acción (novela) se tomó en cada situación (usuario) y funcionó
historial_exitos = {
    'USUARIO_01_MercadoConMiedo': 'VENDER_FUERTE',  # El miedo causó caída, vender fue correcto
    'USUARIO_02_MercadoOptimista': 'COMPRAR_FUERTE',  # Optimismo llevó a alza, comprar fue correcto
    'USUARIO_03_VolatilidadAlta': 'MANTENER',  # En incertidumbre, esperar fue mejor opción
    'USUARIO_04_RecorteOPEP': 'COMPRAR_FUERTE',  # Recorte subió precios, comprar fue acertado
    'USUARIO_05_CrisisRecesion': 'VENDER_MODERADO',  # Recesión bajó demanda, vender fue prudente
    'USUARIO_06_AltaDemandaVerano': 'COMPRAR_MODERADO',  # Demanda alta empujó precios, comprar fue bueno
    'USUARIO_07_ColapsoPrecio': 'COBERTURA'  # Colapso requirió protección, hedging fue necesario
}

print("\n✓ Historial de éxitos registrado:\n")
for usuario, novela in historial_exitos.items():
    contexto = usuarios_historicos[usuario]['descripcion']
    accion = novelas_disponibles[novela]['accion']
    print(f"  • {usuario}")
    print(f"    Situación: {contexto}")
    print(f"    Acción exitosa: {accion} ({novela})\n")

# ══════════════════════════════════════════════════════════════════════════════
# PARTE 4: CARACTERIZACIÓN DE LA SITUACIÓN ACTUAL
# ══════════════════════════════════════════════════════════════════════════════

print("="*90)
print("PARTE 4: SITUACIÓN ACTUAL DEL MERCADO")
print("="*90)

"""
Ahora caracterizamos la situación ACTUAL del mercado.

En un sistema real, estos valores vendrían de:
   - APIs de precios (Yahoo Finance, Bloomberg)
   - Análisis de sentimiento de noticias (VADER, GPT)
   - Indicadores técnicos (RSI, MACD, tendencias)
   - Datos de inventarios (EIA, API)

Para este ejemplo didáctico, simulamos una situación.
"""

# EJEMPLO DIDÁCTICO: Mercado actual con noticias negativas
print("\n📊 SITUACIÓN ACTUAL (EJEMPLO):")
print("   Fecha: 2024-12-03")
print("   Contexto: Mercado nervioso por rumores de exceso de oferta")
print("   Noticias: 'Arabia Saudita considera aumentar producción'")
print("   Inventarios USA: Crecieron más de lo esperado")
print("   Sentimiento general: NEGATIVO (-0.45)\n")

# Vector de la situación actual
# [precio_tendencia, volatilidad, sentimiento, demanda, inventarios, riesgo_geo]
situacion_actual = {
    'vector': np.array([0.35, 0.75, 0.25, 0.45, 0.85, 0.70]),
    'descripcion': 'Mercado nervioso con noticias negativas',
    'componentes': {
        'precio_tendencia': 0.35,  # BAJISTA (precio cayendo)
        'volatilidad': 0.75,  # ALTA volatilidad
        'sentimiento': 0.25,  # NEGATIVO (0.25 de 1.0)
        'demanda': 0.45,  # MODERADA demanda
        'inventarios': 0.85,  # ALTOS inventarios (malo para precio)
        'riesgo_geopolitico': 0.70  # ALTO riesgo
    }
}

print("Vector característico de la situación actual:")
print(f"  {situacion_actual['vector']}\n")

print("Desglose:")
for componente, valor in situacion_actual['componentes'].items():
    nivel = "ALTO" if valor > 0.66 else "MEDIO" if valor > 0.33 else "BAJO"
    print(f"  • {componente:20s}: {valor:.2f} ({nivel})")

# ══════════════════════════════════════════════════════════════════════════════
# PARTE 5: CÁLCULO DE SIMILITUD (COSINE SIMILARITY)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*90)
print("PARTE 5: CÁLCULO DE SIMILITUD CON SITUACIONES HISTÓRICAS")
print("="*90)

"""
COSINE SIMILARITY:
   - Mide el ángulo entre dos vectores
   - Resultado: 1.0 = idénticos, 0.0 = sin relación, -1.0 = opuestos
   - NO depende de la magnitud, solo de la DIRECCIÓN/PATRÓN

Fórmula:
                     A · B
   similarity = ─────────────
                 ||A|| × ||B||

¿Por qué Cosine y no Euclidean?
   - Euclidean penaliza diferencias de magnitud
   - Cosine solo mira el PATRÓN (lo que importa en mercados)
"""

print("\n🔍 Comparando situación actual con todas las situaciones históricas...\n")

# Almacenar similitudes
similitudes = {}

vector_actual = situacion_actual['vector'].reshape(1, -1)

for usuario_id, datos in usuarios_historicos.items():
    vector_historico = datos['vector'].reshape(1, -1)
    
    # CALCULAR COSINE SIMILARITY
    similitud = cosine_similarity(vector_actual, vector_historico)[0][0]
    
    similitudes[usuario_id] = {
        'similitud': similitud,
        'descripcion': datos['descripcion'],
        'contexto': datos['contexto'],
        'vector': datos['vector']
    }

# Ordenar por similitud (de mayor a menor)
similitudes_ordenadas = sorted(similitudes.items(), key=lambda x: x[1]['similitud'], reverse=True)

# Mostrar tabla de resultados
print(f"{'Usuario Histórico':<35} {'Similitud':<12} {'Descripción'}")
print("─"*90)

for usuario_id, datos in similitudes_ordenadas:
    sim = datos['similitud']
    desc = datos['descripcion'][:45]
    print(f"{usuario_id:<35} {sim:>10.4f}  {desc}")

# ══════════════════════════════════════════════════════════════════════════════
# PARTE 6: SELECCIÓN DE RECOMENDACIÓN
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*90)
print("PARTE 6: GENERACIÓN DE RECOMENDACIÓN")
print("="*90)

"""
LÓGICA DEL SISTEMA:
   1. Encontrar la situación histórica MÁS SIMILAR (mayor Cosine Similarity)
   2. Ver qué acción (novela) fue exitosa en ESA situación
   3. Recomendar LA MISMA acción para la situación actual

Este es el principio de los sistemas de recomendación basados en similitud.
"""

# Encontrar el usuario (situación) más similar
usuario_mas_similar_id, datos_similar = similitudes_ordenadas[0]
similitud_maxima = datos_similar['similitud']

# Buscar qué acción (novela) fue exitosa en esa situación
novela_recomendada_id = historial_exitos[usuario_mas_similar_id]
novela_recomendada = novelas_disponibles[novela_recomendada_id]

print("\n🎯 RESULTADO DEL ANÁLISIS:\n")
print(f"Situación histórica más parecida:")
print(f"  ID: {usuario_mas_similar_id}")
print(f"  Descripción: {datos_similar['descripcion']}")
print(f"  Contexto: {datos_similar['contexto']}")
print(f"  Similitud (Cosine): {similitud_maxima:.4f} (escala 0.0-1.0)")

print(f"\nEn esa situación histórica, la acción exitosa fue:")
print(f"  Novela: {novela_recomendada_id}")
print(f"  Acción: {novela_recomendada['accion']}")
print(f"  Nivel: {novela_recomendada['nivel']}")
print(f"  Explicación: {novela_recomendada['explicacion']}")

print("\n" + "─"*90)
print("                        RECOMENDACIÓN FINAL")
print("─"*90)

print(f"\n  🎬 ACCIÓN RECOMENDADA: {novela_recomendada['accion']}")
print(f"  📊 Nivel de convicción: {novela_recomendada['nivel']}")
print(f"  ⚠️  Nivel de riesgo: {novela_recomendada['riesgo']}")
print(f"  ⏱️  Horizonte temporal: {novela_recomendada['horizonte']}")

print(f"\n  💡 JUSTIFICACIÓN:")
print(f"     La situación actual (mercado nervioso con noticias negativas)")
print(f"     tiene un patrón MUY SIMILAR (similitud={similitud_maxima:.2f}) a:")
print(f"     '{datos_similar['descripcion']}'")
print(f"     ")
print(f"     En esa situación histórica, la acción '{novela_recomendada['accion']}'")
print(f"     resultó exitosa. Por lo tanto, recomendamos la misma acción HOY.")

# ══════════════════════════════════════════════════════════════════════════════
# PARTE 7: VISUALIZACIONES
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*90)
print("PARTE 7: GENERANDO VISUALIZACIONES")
print("="*90)

# Gráfico 1: Comparación de vectores (actual vs más similar)
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Subplot 1: Comparación de vectores
ax1 = axes[0]
categorias = ['Precio↗', 'Volatilidad', 'Sentimiento', 'Demanda', 'Inventarios', 'Riesgo Geo']
x = np.arange(len(categorias))
width = 0.35

vector_actual_plot = situacion_actual['vector']
vector_similar_plot = datos_similar['vector']

ax1.bar(x - width/2, vector_actual_plot, width, label='Situación Actual', color='steelblue')
ax1.bar(x + width/2, vector_similar_plot, width, label=f'Histórico Más Similar\n({usuario_mas_similar_id})', color='coral')

ax1.set_ylabel('Valor Normalizado', fontsize=12)
ax1.set_title(f'Comparación: Situación Actual vs Histórico Más Similar (Cosine Sim: {similitud_maxima:.4f})', 
              fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(categorias, fontsize=10)
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, 1.0])

# Subplot 2: Ranking de similitudes
ax2 = axes[1]
usuarios_ids = [uid.replace('USUARIO_', '').replace('_', ' ') for uid, _ in similitudes_ordenadas]
sim_values = [datos['similitud'] for _, datos in similitudes_ordenadas]
colors = ['green' if s > 0.8 else 'orange' if s > 0.6 else 'gray' for s in sim_values]

ax2.barh(usuarios_ids, sim_values, color=colors)
ax2.set_xlabel('Cosine Similarity', fontsize=12)
ax2.set_title('Ranking de Similitud: Todas las Situaciones Históricas', fontsize=14, fontweight='bold')
ax2.set_xlim([0, 1.0])
ax2.grid(axis='x', alpha=0.3)

# Añadir línea vertical en el valor máximo
ax2.axvline(x=similitud_maxima, color='red', linestyle='--', linewidth=2, label=f'Máxima similitud: {similitud_maxima:.4f}')
ax2.legend()

plt.tight_layout()
ruta_grafico = f"{DIRECTORIO_RESULTADOS}/analisis_similitud.png"
plt.savefig(ruta_grafico, dpi=200, bbox_inches='tight')
print(f"\n✓ Gráfico guardado: {ruta_grafico}")
plt.close()

# Gráfico 2: Mapa de calor de similitudes
fig, ax = plt.subplots(figsize=(10, 8))

# Crear matriz de vectores para visualizar
matriz_vectores = []
labels_usuarios = []

# Agregar situación actual primero
matriz_vectores.append(situacion_actual['vector'])
labels_usuarios.append('ACTUAL')

# Agregar todos los históricos
for usuario_id, datos in usuarios_historicos.items():
    matriz_vectores.append(datos['vector'])
    labels_usuarios.append(usuario_id.replace('USUARIO_', '').replace('_', '\n'))

matriz_vectores = np.array(matriz_vectores)

sns.heatmap(matriz_vectores.T, annot=True, fmt='.2f', cmap='YlOrRd', 
            xticklabels=labels_usuarios, 
            yticklabels=categorias,
            cbar_kws={'label': 'Valor Normalizado'},
            ax=ax)

ax.set_title('Mapa de Calor: Características de Todas las Situaciones', fontsize=14, fontweight='bold')
ax.set_xlabel('Situaciones del Mercado', fontsize=12)
ax.set_ylabel('Características', fontsize=12)

plt.tight_layout()
ruta_mapa = f"{DIRECTORIO_RESULTADOS}/mapa_calor_situaciones.png"
plt.savefig(ruta_mapa, dpi=200, bbox_inches='tight')
print(f"✓ Mapa de calor guardado: {ruta_mapa}")
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PARTE 8: GUARDAR REPORTE FINAL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*90)
print("PARTE 8: GUARDANDO REPORTE FINAL")
print("="*90)

reporte = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SISTEMA DE RECOMENDACIÓN DE PETRÓLEO                      ║
║                          REPORTE DE ANÁLISIS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

📅 Fecha del análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🏛️  Instituto: TECSUP Arequipa, Perú

═══════════════════════════════════════════════════════════════════════════════

1. SITUACIÓN ACTUAL DEL MERCADO

Descripción: {situacion_actual['descripcion']}

Vector característico:
  {situacion_actual['vector']}

Desglose de componentes:
"""

for componente, valor in situacion_actual['componentes'].items():
    nivel = "ALTO" if valor > 0.66 else "MEDIO" if valor > 0.33 else "BAJO"
    reporte += f"  • {componente:25s}: {valor:.2f} ({nivel})\n"

reporte += f"""
═══════════════════════════════════════════════════════════════════════════════

2. ANÁLISIS DE SIMILITUD (COSINE SIMILARITY)

Se comparó la situación actual con {len(usuarios_historicos)} situaciones históricas.

SITUACIÓN HISTÓRICA MÁS SIMILAR:
  ID: {usuario_mas_similar_id}
  Descripción: {datos_similar['descripcion']}
  Contexto: {datos_similar['contexto']}
  
  Similitud (Cosine): {similitud_maxima:.4f}
  (Escala: 1.0 = idénticos, 0.0 = sin relación)

═══════════════════════════════════════════════════════════════════════════════

3. RECOMENDACIÓN FINAL

🎬 ACCIÓN RECOMENDADA: {novela_recomendada['accion']}

Detalles:
  • Nivel de agresividad: {novela_recomendada['nivel']}
  • Explicación: {novela_recomendada['explicacion']}
  • Nivel de riesgo: {novela_recomendada['riesgo']}
  • Horizonte temporal: {novela_recomendada['horizonte']}

═══════════════════════════════════════════════════════════════════════════════

4. JUSTIFICACIÓN

La situación actual del mercado presenta un patrón muy similar
(similitud = {similitud_maxima:.2f}) a la situación histórica:

  "{datos_similar['descripcion']}"

Ocurrida el {usuarios_historicos[usuario_mas_similar_id]['fecha']}, en la cual:
  {datos_similar['contexto']}

En esa situación, la acción que resultó exitosa fue:
  **{novela_recomendada['accion']}** ({novela_recomendada_id})

Por lo tanto, basándonos en el principio de similitud de patrones,
recomendamos tomar LA MISMA ACCIÓN en la situación actual.

═══════════════════════════════════════════════════════════════════════════════

5. RANKING COMPLETO DE SIMILITUDES

(Ordenado de mayor a menor similitud)

"""

for i, (uid, datos) in enumerate(similitudes_ordenadas, 1):
    reporte += f"{i}. {uid}\n"
    reporte += f"   Similitud: {datos['similitud']:.4f}\n"
    reporte += f"   Descripción: {datos['descripcion']}\n"
    reporte += f"   Acción histórica: {historial_exitos[uid]}\n\n"

reporte += """
═══════════════════════════════════════════════════════════════════════════════

6. METODOLOGÍA: ¿POR QUÉ COSINE SIMILARITY?

Cosine Similarity mide el ÁNGULO entre vectores, no su magnitud.

Ventajas para este sistema:
  ✓ Compara PATRONES de mercado, no valores absolutos
  ✓ Funciona bien con datos normalizados [0,1]
  ✓ No penaliza diferencias en magnitud
  ✓ Estándar en sistemas de recomendación (Netflix, Amazon)
  ✓ Eficiente computacionalmente

Fórmula:
                     A · B
   similarity = ─────────────
                 ||A|| × ||B||

═══════════════════════════════════════════════════════════════════════════════

7. ADVERTENCIAS Y LIMITACIONES

  ⚠️  Este sistema se basa en patrones históricos. Eventos sin precedentes
      (crisis COVID-19, guerras) pueden generar predicciones incorrectas.

  ⚠️  La recomendación es una ORIENTACIÓN, no asesoría financiera profesional.

  ⚠️  Siempre considere factores adicionales: análisis fundamental, noticias
      recientes, variables macroeconómicas, riesgo personal.

═══════════════════════════════════════════════════════════════════════════════

FIN DEL REPORTE
"""

# Guardar reporte
ruta_reporte = f"{DIRECTORIO_RESULTADOS}/reporte_recomendacion.txt"
with open(ruta_reporte, 'w', encoding='utf-8') as f:
    f.write(reporte)

print(f"\n✓ Reporte completo guardado: {ruta_reporte}")

# ══════════════════════════════════════════════════════════════════════════════
# PARTE 9: RESUMEN FINAL EN TERMINAL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*90)
print("                            RESUMEN FINAL")
print("="*90)

print(f"\n✅ Sistema ejecutado exitosamente\n")
print(f"📊 Situación actual: {situacion_actual['descripcion']}")
print(f"🔍 Situación más similar: {datos_similar['descripcion']}")
print(f"📈 Similitud (Cosine): {similitud_maxima:.4f}")
print(f"\n🎬 RECOMENDACIÓN: {novela_recomendada['accion']} ({novela_recomendada['nivel']})")
print(f"⚠️  Riesgo: {novela_recomendada['riesgo']}")
print(f"⏱️  Horizonte: {novela_recomendada['horizonte']}")

print(f"\n📁 Archivos generados en '{DIRECTORIO_RESULTADOS}/':")
print(f"   • reporte_recomendacion.txt")
print(f"   • analisis_similitud.png")
print(f"   • mapa_calor_situaciones.png")

print("\n" + "="*90)
print("¡SISTEMA COMPLETADO!")
print("="*90 + "\n")
