import pandas as pd
import numpy as np

# Parámetros de ponderación: 
# - peso_seleccion: peso para la parte de selección (por ejemplo, si la respuesta indica importancia)
# - peso_ranking: peso para la parte de ranking (puntaje calculado según orden de preferencia)
peso_seleccion = 0.3
peso_ranking = 0.7

# Cargar el archivo CSV
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
archivo_csv = BASE_DIR / "data" / "survey_sample.csv"
df = pd.read_csv(archivo_csv, encoding='latin1', sep=',', low_memory=False)

# Definir temas y DataFrame de resultados
variables = [
    "swd",            # aceras delimitadas
    "ws",             # ancho aceras
    "pedFlow",        # Flujo peatonal
    "pvQ",            # Calidad pavimento
    "pedInfr",        # Infraestructura peatonal
    "Obstr",          # Obstáculos en acera
    "Pres_pol_c",     # Presencia de elementos de seguridad (policías, cámaras)
    "Crime",          # Crimen
    "Crashes",        # Accidentes
    "Lighting",       # Iluminación
    "Cs",             # Velocidad vehicular
    "ptStops",        # Paradas de buses
    "Tcontrl",        # Control de tráfico
    "trFlow",         # Flujo vehicular
    "crTime",         # Tiempo de cruce
    "trDens",         # Suelo comercial
    "insDens",        # Suelo institucional/administrativo
    "ResDens",        # Suelo residencial
    "Aesthetics",     # Estética de edificios
    "Trees",          # Presencia de árboles
    "Block",          # Largo de cuadras
    "noise",          # Nivel de ruido
    "StagH2o",        # Agua estancada
    "Slope",          # Pendiente
    "AirPol",         # Contaminación
    "Cleanless",      # Limpieza en las calles
    "mpgaDens",       # Areas verdes
    "design",         # Tópicos: (design-atract)
    "safety",
    "security",
    "destin",
    "atract"
]
df_resultados = pd.DataFrame(0.0, index=df.index, columns=variables)

# Mapear preguntas a variables
column_to_variable41 = {
    "DES_DISP1": "swd", "DES_DISP2": "ws", "DES_DISP3": "pvQ",
    "DES_DISP4": "pedInfr", "DES_DISP5": "Obstr"
}
column_to_variable42 = {
    "R_DES1": "swd", "R_DES2": "ws", "R_DES3": "pvQ",
    "R_DES4": "pedInfr", "R_DES5": "Obstr"
}
column_to_variable51 = {
    "SAF_DISP1": "Pres_pol_c", "SAF_DISP2": "pedFlow",
    "SAF_DISP3": "Crime", "SAF_DISP4": "Lighting"
}
column_to_variable52 = {
    "R_SAF1": "Pres_pol_c", "R_SAF2": "pedFlow",
    "R_SAF3": "Crime", "R_SAF4": "Lighting"
}
column_to_variable61 = {
    "SEC_DISP1": "Cs", "SEC_DISP2": "trFlow", "SEC_DISP3": "Tcontrl", 
    "SEC_DISP4": "Crashes", "SEC_DISP5": "crTime"
}
column_to_variable62 = {
    "R_SEC1": "Cs", "R_SEC2": "trFlow", "R_SEC3": "Tcontrl", 
    "R_SEC4": "Crashes", "R_SEC5": "crTime"
}
column_to_variable71 = {
    "DEST_DISP1": "trDens", "DEST_DISP2": "insDens", "DEST_DISP3": "ResDens", 
    "DEST_DISP4": "ptStops", "DEST_DISP5": "mpgaDens"
}
column_to_variable72 = {
    "R_DEST1": "trDens", "R_DEST2": "insDens", "R_DEST3": "ResDens", 
    "R_DEST4": "ptStops", "R_DEST5": "mpgaDens"
}
column_to_variable81 = {
    "ATR_DISP1": "Trees", "ATR_DISP2": "Aesthetics", "ATR_DISP3": "noise",
    "ATR_DISP4": "Cleanless", "ATR_DISP5": "StagH2o", "ATR_DISP6": "AirPol",
    "ATR_DISP7": "Slope", "ATR_DISP8": "Block"
}
column_to_variable82 = {
    "R_ATR1": "Trees", "R_ATR2": "Aesthetics", "R_ATR3": "noise", 
    "R_ATR4": "Cleanless", "R_ATR5": "StagH2o", "R_ATR6": "AirPol",
    "R_ATR7": "Slope", "R_ATR8": "Block"
}

# Función vectorizada para calcular los puntajes usando el ranking dado por el encuestado
def calcular_puntuaciones(df, column_to_variable, column_prefix, df_resultados, num_columnas):
    col_names = [f"{column_prefix}{i}" for i in range(1, num_columnas+1)]
    # Reemplazar 999999 por NaN para identificar las respuestas válidas
    sub = df[col_names].replace(999999, np.nan).astype(float)
    # Para cada fila, n es el máximo valor (ranking) entre las respuestas válidas
    n = sub.max(axis=1)
    # Denominador para normalizar: n*(n+1)/2 (serie)
    denominator = n * (n + 1) / 2
    # Convertir Series a arrays columnares para evitar problemas de broadcasting
    n_arr = n.values.reshape(-1, 1)
    # Reemplazar ceros en el denominador para evitar división por cero
    denom_arr = np.where(denominator == 0, 1, denominator).reshape(-1, 1)
    # Realizar la operación en numpy para obtener una matriz de puntajes (n_filas x num_columnas)
    score_arr = (n_arr - (sub.values - 1)) / denom_arr

    score_arr = np.where(np.isnan(score_arr), 0, score_arr)
    score_df = pd.DataFrame(score_arr, index=sub.index, columns=sub.columns)
    # Sumar los puntajes a las variables correspondientes, multiplicando por el peso del ranking
    for col in col_names:
        if col in column_to_variable:
            var = column_to_variable[col]
            df_resultados[var] += peso_ranking * score_df[col]

def calcular_puntuaciones_2(df, column_to_variable, column_names, df_resultados):
    # Reemplazar 999999 por NaN para identificar respuestas válidas
    sub = df[column_names].replace(999999, np.nan).astype(float)
    n = sub.max(axis=1)
    denominator = n * (n + 1) / 2
    n_arr = n.values.reshape(-1, 1)
    denom_arr = np.where(denominator == 0, 1, denominator).reshape(-1, 1)
    score_arr = (n_arr - (sub.values - 1)) / denom_arr
    score_arr = np.where(np.isnan(score_arr), 0, score_arr)
    score_df = pd.DataFrame(score_arr, index=sub.index, columns=sub.columns)
    # Sumar los puntajes a las variables correspondientes, con el peso del ranking
    for col in column_names:
        if col in column_to_variable:
            var = column_to_variable[col]
            df_resultados[var] += peso_ranking * score_df[col]

# Procesar Q4.1 (selección: se multiplica por peso_seleccion)
for col, var in column_to_variable41.items():
    if col in df.columns:
        df_resultados[var] += peso_seleccion * df[col].fillna(0).astype(float)

# Procesar Q4.2 (ranking vectorizado)
calcular_puntuaciones(df, column_to_variable42, "R_DES", df_resultados, num_columnas=5)

# Procesar Q5.1 (selección)
for col, var in column_to_variable51.items():
    if col in df.columns:
        df_resultados[var] += peso_seleccion * df[col].fillna(0).astype(float)

# Procesar Q5.2 (ranking vectorizado)
calcular_puntuaciones(df, column_to_variable52, "R_SAF", df_resultados, num_columnas=4)

# Q6.1 (selección)
for col, var in column_to_variable61.items():
    if col in df.columns:
        df_resultados[var] += peso_seleccion * df[col].fillna(0).astype(float)
# Procesar Q6.2 (ranking vectorizado)
calcular_puntuaciones(df, column_to_variable62, "R_SEC", df_resultados, num_columnas=5)

# Q7.1 (selección)
for col, var in column_to_variable71.items():
    if col in df.columns:
        df_resultados[var] += peso_seleccion * df[col].fillna(0).astype(float)

# Q7.2 (ranking vectorizado)
calcular_puntuaciones(df, column_to_variable72, "R_DEST", df_resultados, num_columnas=5)

# Q8.1 (selección)
for col, var in column_to_variable81.items():
    if col in df.columns:
        df_resultados[var] += peso_seleccion * df[col].fillna(0).astype(float)
        
# Q8.2 (ranking vectorizado)
calcular_puntuaciones(df, column_to_variable82, "R_ATR", df_resultados, num_columnas=8)

# Q9 (ranking vectorizado; no hay parte de selección en esta pregunta)
column_to_variable92 = {
    "R_DESIGN": "design", "R_SAFETY": "safety", "R_SECURITY": "security",
    "R_DESTIN": "destin", "R_ATRACT": "atract"
}
columnas_pregunta = ["R_DESIGN", "R_SAFETY", "R_SECURITY", "R_DESTIN", "R_ATRACT"]
calcular_puntuaciones_2(df, column_to_variable92, columnas_pregunta, df_resultados)

# Preugnta 10. Edad

df_resultados["AGE"] = df["AGE"]


# Pregunta 11. Género

df_resultados.loc[:, "GENDER"] = df["GENDER"].map({"Masculino": 0, "Femenino": 1, "Otro": 2})

# Pregunta 12. Discapacidad

# Pregunta 12. Discapacidad

# Lista de tipos de discapacidad con sus nombres de columna
tipos_discapacidad = {
    "Dificultad para caminar o moverme libremente.": "Discapacidad_Motora",
    "Dificultad para oír": "Discapacidad_Auditiva",
    "Dificultad para ver.": "Discapacidad_Visual",
    "Dificultad para hablar o comunicarme.": "Discapacidad_Comunicacion"
}

# Inicializar las columnas con 0
for col in tipos_discapacidad.values():
    df_resultados[col] = 0

# Asegurar que la columna DISABILITY sea de tipo string y manejar NaN
df["DISABILITY"] = df["DISABILITY"].astype(str)  # Convertir todo a string

# Separar los resultados múltiples
df_resultados["DISABILITY_split"] = df["DISABILITY"].apply(
    lambda x: x.split(",") if "," in x else x.split(";") if ";" in x else [x]
)

# Función para asignar 1 si corresponde a cada tipo de discapacidad
def asignar_discapacidad(resultados):
    # Si se encuentra "No presento ninguna dificultad o limitación.", devolver 0 para todo
    if "No presento ninguna dificultad o limitación." in resultados:
        return {col: 0 for col in tipos_discapacidad.values()}
    
    # Si no, asignar 1 a las discapacidades correspondientes
    return {
        col: 1 if any(texto.strip() in respuesta.strip() for respuesta in resultados) else 0
        for texto, col in tipos_discapacidad.items()
    }

# Aplicar la función y asignar los resultados a las columnas
discapacidades_asignadas = df_resultados["DISABILITY_split"].apply(asignar_discapacidad)

# Convertir el resultado en un DataFrame y asignarlo a df_resultados
discapacidades_asignadas_df = pd.DataFrame(discapacidades_asignadas.tolist(), index=df_resultados.index)
df_resultados[discapacidades_asignadas_df.columns] = discapacidades_asignadas_df

# Columna general para indicar si tiene alguna discapacidad
df_resultados["DISABILITY"] = df_resultados[list(tipos_discapacidad.values())].max(axis=1)


# Pregunta 13 Q13

# Diccionario de rangos de ingresos
rango_ingresos = {
    "0 - 400.000": 1,
    "400.000 - 800.000": 2,
    "800.000 - 1.200.000": 3,
    "1.200.000 - 1.600.000": 4,
    "1.600.000 - 2.000.000": 5,
    "Más de 2.000.000": 6
}

# Normalizar el texto en Q13
df_resultados["INC"] = df["INC"].astype(str).str.strip()  # Convertir a string y eliminar espacios
df_resultados["INC"] = df_resultados["INC"].str.replace("\u200b", "", regex=True)  # Eliminar caracteres ocultos
df_resultados["INC"] = df_resultados["INC"].str.replace(",", ".", regex=True)  # Normalizar separadores

# Crear nueva columna con los valores numéricos
df_resultados["INC"] = df_resultados["INC"].map(rango_ingresos)

# Pregunta 14. En qué comuna reside?

# Diccionario de mapeo de comunas a números
mapeo_comunas = {
    "Chiguayante": 1,
    "Concepción": 2,
    "Coronel": 3,
    "Florida": 4,
    "Hualpén": 5,
    "Hualqui": 6,
    "Lota": 7,
    "Penco": 8,
    "San Pedro de la Paz": 9,
    "Santa Juana": 10,
    "Talcahuano": 11,
    "Tomé": 12
}

# Preprocesar la columna NEIGH del df
df_resultados["NEIGH"] = df["NEIGH"].astype(str).str.strip()  # Eliminar espacios al inicio y final

# Crear la columna con los valores numéricos
df_resultados["NEIGH"] = df_resultados["NEIGH"].map(mapeo_comunas)


# Pregunta 14.1. En que barrio de la comuna

# Diccionario de mapeo de barrios a números
mapeo_barrios = {
    "Agüita de la periz": 1,
    "Almirante rivera norte": 2,
    "Barrio norte": 3,
    "Brisas del sol": 4,
    "Cerro David Fuentes": 5,
    "Cerro San Francisco": 6,
    "Cerro verde": 7,
    "Colón 9000": 8,
    "Concepción centro": 9,
    "Lan B": 10,
    "Lan c": 11,
    "Las Américas": 12,
    "Lorenzo arenas": 13,
    "Los boldoa": 14,  # Unificado con "Los boldos"
    "Los boldos": 14,  # Unificado con "Los boldoa"
    "Palomares": 15,
    "Parque invicoop": 16,  # Unificado con "Paruqe invicoop"
    "Paruqe invicoop": 16,  # Unificado con "Parque invicoop"
    "Pedro de Valdivia": 17,
    "Presidente bulnes": 18,
    "San rosendo": 19,
    "San Vicente": 20,
    "Villa Acero": 21,
    "Villa san rosendo": 22,
    "Otro": 23,
    "999999": np.nan  # Valor nulo
}

# Preprocesar la columna NEIGH2 del df
df_resultados["NEIGH2"] = df["NEIGH2"].astype(str).str.strip()  # Eliminar espacios al inicio y final

# Unificar variantes de nombres
df_resultados["NEIGH2"] = df_resultados["NEIGH2"].replace({
    "Los boldos": "Los boldoa",  # Unificar "Los boldos" con "Los boldoa"
    "Paruqe invicoop": "Parque invicoop"  # Unificar "Paruqe invicoop" con "Parque invicoop"
})

# Crear la columna con los valores numéricos
df_resultados["NEIGH2"] = df_resultados["NEIGH2"].map(mapeo_barrios)


# Pregunta 15. Ocupación

# Diccionario de mapeo de ocupacion a números
mapeo_ocupacion = {
    "Empleado": 1,
    "Comerciante": 2,
    "Independiente": 3,
    "Estudiante": 4,
    "Estudiante y empleado": 5,
    "Actividades del hogar": 6,
    "Desempleado": 7,
    "Pensionado": 8,
    "Otro": 9
}

# Preprocesar la columna Q15
df_resultados["OCCUP"] = df["OCCUP"].astype(str).str.strip()  # Eliminar espacios al inicio y final

# Crear la columna "ocupacion_cod" con los valores numéricos
df_resultados["OCCUP"] = df_resultados["OCCUP"].map(mapeo_ocupacion)


# Pregunta 16. EDUCATION

# Diccionario de mapeo de nivel educativo a números
mapeo_educacion = {
    "Escuela primaria": 1,
    "Escuela secundaria": 2,
    "Educación técnica o tecnológica": 3,
    "Pregrado universitario": 4,
    "Posgrado universitario": 5,
    "Ninguna de las anteriores": 6
}

# Preprocesar la columna Q16
df_resultados["EDUCATION"] = df["EDUCATION"].astype(str).str.strip()  # Eliminar espacios al inicio y final

# Crear la columna "educacion_cod" con los valores numéricos
df_resultados["EDUCATION"] = df_resultados["EDUCATION"].map(mapeo_educacion)



# Pregunta 17 Viajes a pie DIA LABORAL típico

# Reemplazamos '5 o más' por 5 en la columna 'NWT_WD'
df_resultados['NWT_WD'] = df['NWT_WD'].replace('5 o más', 5)
# Convertimos la columna a valores numéricos, para asegurar que todo se maneje correctamente
df_resultados['NWT_WD'] = pd.to_numeric(df['NWT_WD'], errors='coerce')


# Pregunta 18. Duración de los viajes a pia DIA LABORAL


# Diccionario de mapeo de duración de viajes a números
mapeo_duracion = {
    "Menos de 5 min por viaje.": 1,
    "Entre 5 a 10 min por viaje.": 2,
    "Entre 10 a 15 min por viaje.": 3,
    "Entre 15 a 20 min por viaje.": 4,
    "Entre 20 a 30 min por viaje.": 5,
    "Entre 30 a 45 min por viaje.": 6,
    "Entre 45 a 60 min por viaje.": 7,
    "Más de 60 min por viaje.": 8,
    "No realizo viajes a pie.": 0  # Asignamos 0 para indicar que no hay viajes
}

# Supongamos que la columna con las resultados se llama DWT_WD
df_resultados["DWT_WD"] = df["DWT_WD"].astype(str).str.strip()  # Limpiar espacios

# Crear la columna "Duracion_viaje_cod" con los valores numéricos
df_resultados["DWT_WD"] = df_resultados["DWT_WD"].map(mapeo_duracion)


# Pregunta 19. En qué sector o barrio realiza estos viajes a pie? DIA LABORAL

# Diccionario de mapeo de comunas a números
mapeo_comunas = {
    "Chiguayante": 1,
    "Concepción": 2,
    "Coronel": 3,
    "Florida": 4,
    "Hualpén": 5,
    "Hualqui": 6,
    "Lota": 7,
    "Penco": 8,
    "San Pedro de la Paz": 9,
    "Santa Juana": 10,
    "Talcahuano": 11,
    "Tomé": 12
}

# Preprocesar la columna NEIGH_WD del df
df_resultados["NEIGH_WD"] = df["NEIGH_WD"].astype(str).str.strip()  # Eliminar espacios al inicio y final

# Crear la columna con los valores numéricos
df_resultados["NEIGH_WD"] = df_resultados["NEIGH_WD"].map(mapeo_comunas)


# Pregunta 19.1. Barrio donde camina

# 🐙🐙🐙🐖🐖🐻‍❄️🤍🤍 MODELAR DESPUES 🐙🐙🐙🐖🐖🐻‍❄️🤍🤍


# Pregunta 20. Viajes a pie DIA NO LABORAL típico

# Reemplazamos '5 o más' por 5 en la columna 'NWT_WE'
df_resultados['NWT_WE'] = df['NWT_WE'].replace('5 o más', 5)
# Convertimos la columna a valores numéricos, para asegurar que todo se maneje correctamente
df_resultados['NWT_WE'] = pd.to_numeric(df['NWT_WE'], errors='coerce')



# Pregunta 21. Duración de los viajes a pie DIA NO LABORAL

# Diccionario de mapeo de duración de viajes a números
mapeo_duracion = {
    "Menos de 5 min por viaje.": 1,
    "Entre 5 a 10 min por viaje.": 2,
    "Entre 10 a 15 min por viaje.": 3,
    "Entre 15 a 20 min por viaje.": 4,
    "Entre 20 a 30 min por viaje.": 5,
    "Entre 30 a 45 min por viaje.": 6,
    "Entre 45 a 60 min por viaje.": 7,
    "Más de 60 min por viaje.": 8,
    "No realizo viajes a pie.": 0  # Asignamos 0 para indicar que no hay viajes
}

# Supongamos que la columna con las resultados se llama DWT_WE
df_resultados["DWT_WE"] = df["DWT_WE"].astype(str).str.strip()  # Limpiar espacios

# Crear la columna "Duracion_viaje_cod" con los valores numéricos
df_resultados["DWT_WE"] = df_resultados["DWT_WE"].map(mapeo_duracion)

# Pregunta 22. En qué sector o barrio realiza estos viajes a pie? DIA LABORAL

# Diccionario de mapeo de comunas a números
mapeo_comunas = {
    "Chiguayante": 1,
    "Concepción": 2,
    "Coronel": 3,
    "Florida": 4,
    "Hualpén": 5,
    "Hualqui": 6,
    "Lota": 7,
    "Penco": 8,
    "San Pedro de la Paz": 9,
    "Santa Juana": 10,
    "Talcahuano": 11,
    "Tomé": 12
}

# Preprocesar la columna NEIGH_WE del df
df_resultados["NEIGH_WE"] = df["NEIGH_WE"].astype(str).str.strip()  # Eliminar espacios al inicio y final

# Crear la columna con los valores numéricos
df_resultados["NEIGH_WE"] = df_resultados["NEIGH_WE"].map(mapeo_comunas)

# Pregunta 22.1 En que barrio de la pregunta 22

# Diccionario de mapeo de barrios por categoría urbana
mapeo_barrios = {
    # Centro y Áreas Mixtas
    "Concepción centro": 1,
    "Centro": 1,
    "Barrio universitario, barrio oriente, chillancito, parque ecuador, tres pascualas, costanera bio bio, concepción": 1,

    # Barrios Residenciales Consolidados
    "Barrio norte": 2,
    "Lorenzo arenas": 2,
    "Las Américas": 2,
    "Lan B": 2,
    "Lan C": 2,
    "San Vicente": 2,
    "Villa Acero": 2,
    "Villa San rosendo": 2,
    "San Marcos": 2,

    # Barrios en Zonas Altas (Cerros)
    "Cerro David Fuentes": 3,
    "Cerro San Francisco": 3,
    "Cerro verde": 3,
    "Cerros": 3,

    # Barrios Periféricos o en Expansión
    "Arenal": 4,
    "Arenal/san Vicente": 4,
    "Colón 9000": 4,
    "Medio camino": 4,
    "Huachicop": 4,
    "Parque invicoop": 4,

    # Zona Industrial, Portuaria y Mixta
    "Talcahuano": 5,
    "Talcahuano plano": 5,
    "Thno plano": 5,
    "Plano": 5,
    "Puerto": 5,

    # Otros o No Clasificados
    "Los boldos": 6,
    "Otro": 6,
    "999999": np.nan  # Valor nulo
}

# Preprocesar la columna NEIGH2 del df
df_resultados["NEIGH2"] = df["NEIGH2"].astype(str).str.strip()  # Eliminar espacios al inicio y final

# Unificar variantes de nombres
df_resultados["NEIGH2"] = df_resultados["NEIGH2"].replace({
    "Los boldos": "Los boldoa",  # Unificar "Los boldos" con "Los boldoa"
    "Paruqe invicoop": "Parque invicoop"  # Unificar "Paruqe invicoop" con "Parque invicoop"
})

# Crear la columna con los valores numéricos
df_resultados["NEIGH2"] = df_resultados["NEIGH2"].map(mapeo_barrios)



# Guardar resultados
output_csv = BASE_DIR / "outputs" / "processed_survey_sample.csv"
df_resultados.to_csv(output_csv, index=False, sep=',', encoding='latin1', float_format="%.4f")
print(f"Archivo procesado y guardado en: {output_csv}")
