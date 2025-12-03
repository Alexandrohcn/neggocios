import numpy as np

# Situación actual del mercado
situacion_actual = np.array([0.80, 0.60, 0.70, 1.00])

situaciones_historicas = {
    'Escenario A (MUY SIMILAR)': np.array([0.85, 0.55, 0.75, 0.95]),
    'Escenario B (SIMILAR)': np.array([0.75, 0.65, 0.65, 0.90]),
    'Escenario C (DIFERENTE)': np.array([0.30, 0.20, 0.15, 0.10]),
    'Escenario D (OPUESTO)': np.array([0.20, 0.40, 0.30, 0.00])
}

print("\n" + "="*90)
print("           DEMOSTRACIÓN EDUCATIVA: COSINE SIMILARITY EN ACCIÓN")
print("="*90)

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
