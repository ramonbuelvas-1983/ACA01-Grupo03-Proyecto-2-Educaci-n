# TRAYECTORIA-UCI

**Datos educativos y evidencia para la permanencia · Proyecto 2 · ACA 1**

El proyecto integra el CSV entregado de UCI 697 con tres documentos oficiales colombianos sobre bienestar y acompañamiento. Construye siete perfiles descriptivos en SQL, indexa pasajes con embeddings multilingües y devuelve evidencia documental junto con perfiles asociados mediante sus identificadores y una vinculación temática.

Los registros proceden de Portugal. Los documentos conservan el ámbito del Ministerio de Educación Nacional, la Universidad El Bosque y la Universidad Nacional de Colombia; sus orientaciones no se aplican automáticamente a las personas del dataset. El proyecto describe los datos y recupera fuentes, sin entrenar un predictor individual de abandono.

## Entregables

[Ejecutar en Google Colab](https://colab.research.google.com/github/ramonbuelvas-1983/ACA01-Grupo03-Proyecto-2-Educaci-n/blob/main/Trayectoria_UCI_ACA1.ipynb) · [Notebook ejecutado](Trayectoria_UCI_ACA1.ipynb) · [Póster PDF](entregables/Trayectoria_UCI_Poster.pdf) · [Video de sustentación, 4:40](entregables/Trayectoria_UCI_Sustentacion.mp4) · [Subtítulos](entregables/Trayectoria_UCI_Sustentacion.srt)

El notebook registra una ejecución local completa: 15 celdas de código sin errores, 319 pasajes indexados, Hit@3 de 9/12 en desarrollo y 5/6 en reserva. La verificación adicional desde terminal, registrada en `results/cloud_summary.json`, confirma 319 puntos en Qdrant Cloud, el índice `doc_id` de tipo `keyword`, idempotencia y 9/12 aciertos de desarrollo. El notebook y el video conservan su ejecución local; cada archivo de evidencia identifica su backend.

## Ejecutar el notebook

Abra `Trayectoria_UCI_ACA1.ipynb` en Google Colab o Jupyter y ejecute las celdas en orden desde un kernel recién iniciado. El notebook incluye el código, los manifiestos y las dependencias fijadas; si no encuentra la carpeta del proyecto, la reconstruye y descarga las fuentes desde sus enlaces institucionales.

La restauración del CSV verifica que sus bytes coincidan con el SHA-256 del archivo entregado. Cada PDF también tiene una huella de integridad. Si una fuente cambia, la ejecución se detiene para evitar mezclar versiones.

El repositorio conserva el CSV y los manifiestos de las tres fuentes. El notebook descarga los PDF desde sus alojamientos institucionales y comprueba sus hashes; los originales no se redistribuyen en el repositorio. La primera ejecución requiere conexión para instalar dependencias y obtener el modelo de Hugging Face. Las dependencias directas se verificaron en Python 3.12; la primera celda contrasta sus versiones con `requirements-lock.txt` y utiliza el canal CPU cuando necesita instalar PyTorch.

## Ejecutar desde terminal

Desde la carpeta raíz del proyecto:

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m src.prepare_data --root .
python -m src.pipeline --root . --backend local --provider e5 --retrieval rerank
```

La terminal conserva un índice Qdrant local en `results/qdrant_local`; agregue `--memory` para utilizar un almacén transitorio. El notebook emplea la variante transitoria, lo que permite repetirlo sin bloquear el índice persistente utilizado por otras herramientas. Ambas modalidades locales identifican su backend en la evidencia.

## Configurar la ejecución cloud

1. Cree un clúster Free en [Qdrant Cloud](https://cloud.qdrant.io/) y genere una clave con acceso al clúster.
2. Copie `.env.example` como `.env` y complete `QDRANT_URL` y `QDRANT_API_KEY` en su entorno privado.
3. Para el notebook, establezca `RAG_BACKEND=cloud`; para la terminal, utilice el argumento `--backend cloud`.
4. Ejecute la canalización y confirme `remote_verified=True`, dimensión, métrica, conteos y resultados de consulta.

```bash
python -m src.pipeline --root . --backend cloud --provider e5 --retrieval rerank
```

Las credenciales se leen exclusivamente desde variables de entorno o `.env`; nunca deben incluirse en celdas, capturas, salidas o repositorios. Un fallo remoto interrumpe la ejecución e informa su tipo sin mostrar secretos; no se activa un sustituto local. La evidencia de una ejecución local no acredita por sí misma la conexión cloud.

## Herramientas por componente arquitectónico

| Componente | Herramientas utilizadas |
|---|---|
| Ingesta y control del dataset | Python, pandas y SHA-256 |
| Almacenamiento y perfiles tabulares | SQLite y SQL, mediante sqlite3 |
| Extracción y fragmentación documental | PyMuPDF y tokenizador de Transformers |
| Embeddings multilingües | SentenceTransformers, PyTorch y multilingual-e5-small, 384 dimensiones |
| Índice vectorial y consultas | Qdrant Client; Qdrant local y Qdrant Cloud Free en AWS São Paulo |
| Reordenamiento de candidatos | CrossEncoder mmarco-mMiniLMv2-L12-H384-v1 |
| Configuración privada | python-dotenv y variables de entorno |
| Experimentos y visualización | Jupyter/IPython, NumPy y Matplotlib |

El póster se elaboró con ReportLab. El video combina gráficos de resultados, FFmpeg y narración sintética en español generada localmente con Piper.

## Datos, perfiles y corpus

El CSV contiene 4.424 registros, 36 variables predictoras y `Target`. Se forman siete perfiles con tres agrupaciones: unidades curriculares aprobadas en el primer semestre —0, 1–5 y 6 o más—, tenencia de beca —sí/no— y condición de deudor —sí/no—. Los estados educativos se resumen después de formar los perfiles; `Target` no interviene en los embeddings ni en la clasificación de los pasajes.

Cada agrupación cubre el dataset completo, por lo que los perfiles de agrupaciones distintas se superponen. Sus diferencias son descriptivas y no estiman el efecto de becas, deuda o aprobación sobre la permanencia.

| Documento | Año | Páginas físicas |
|---|---:|---:|
| MEN, Lineamientos de política de bienestar para instituciones de educación superior | 2016 | 55 |
| Universidad El Bosque, Política para el éxito estudiantil y lineamientos | 2024 | 48 |
| Universidad Nacional de Colombia, Guía de acompañamiento para estudiantes y tutores de Posgrado, 4.ª edición | 2022 | 51 |

El corpus suma 154 páginas. La copia MEN se descarga desde el alojamiento institucional de la Universidad de San Buenaventura Cali y conserva su origen ministerial en el manifiesto. Las referencias APA 7 figuran en el notebook y los metadatos completos en `data/sources.json`.

## Fragmentación, embeddings y recuperación

La fragmentación conserva párrafos, con máximo de 280 tokens y solapamiento de hasta 50 tokens; los párrafos largos se subdividen con un solapamiento de 40 tokens. Las páginas físicas, las posiciones del texto y los hashes permiten rastrear cada pasaje al documento original.

El modelo principal es `intfloat/multilingual-e5-small`, con revisión fija, 384 dimensiones, prefijos `passage:` y `query:` y normalización L2. La indexación utiliza coseno, lotes de 64 y UUID estables. La caché y la colección dependen de las fuentes y de la configuración del modelo.

La consulta identifica si nombra explícitamente una única institución del corpus —MEN, Universidad El Bosque o Universidad Nacional, con sus alias— y, en ese caso, aplica un filtro por documento. Sin una fuente explícita o con varias instituciones nombradas, consulta todo el corpus. La decisión se registra en `source_scope` y no utiliza las referencias de evaluación.

El modo `rerank` obtiene veinte candidatos dentro del ámbito seleccionado y los reordena con el Cross-Encoder preentrenado `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, utilizando pares de pregunta frente a título y texto del pasaje, sin entrenar sobre el corpus ni utilizar las etiquetas de evaluación durante la búsqueda. `score` conserva el coseno, `dense_rank` su posición inicial y `rerank_score` la puntuación del reordenador; `rank` identifica el orden final. Las dos puntuaciones tienen escalas diferentes y no son probabilidades de exactitud. Seleccione `--retrieval dense` para consultar únicamente por coseno; en el notebook, `RETRIEVAL_METHOD` controla las consultas de demostración. La sección experimental ejecuta ambas configuraciones de forma explícita para compararlas.

El adaptador opcional Gemini Embedding 2 utiliza 768 dimensiones y una colección independiente. Configure `GEMINI_API_KEY` y seleccione `EMBEDDING_PROVIDER=gemini2` en el notebook o `--provider gemini2` en la terminal. El modelo 2 utiliza instrucciones de recuperación dentro del texto, pues no admite `task_type`; se envía una solicitud por pasaje. `GEMINI_REQUEST_INTERVAL` permite ajustar el ritmo a la cuota disponible.

## Evaluación y archivos de evidencia

Se distinguen doce preguntas de desarrollo y seis preguntas de reserva. El conjunto de desarrollo permite comparar la recuperación densa con el reordenamiento, manteniendo el mismo filtro de fuente explícita en ambos métodos; las preguntas de reserva se anotaron sin observar resultados y se utilizan con la configuración fijada, sin ajustar el recuperador ni las etiquetas a partir de ellas. Es una reserva interna pequeña del mismo corpus, no una validación externa.

El sistema calcula Hit@3, MRR@3 y nDCG@3 y contrasta el coseno mediante NumPy. Cada consulta se repite tres veces después del calentamiento: la latencia de desarrollo resume 36 observaciones por configuración y la reserva, 18. La comparación conserva la diferencia de coste de inferencia entre búsqueda densa y Cross-Encoder; las mediciones no representan un acuerdo de nivel de servicio.

La unidad de relevancia exige coincidencia de fuente, página y evidencia textual. Las alternativas de respuesta pertenecen al mismo grupo de evidencia, de modo que recuperar varios fragmentos equivalentes no multiplica la ganancia de nDCG. Las consultas fallidas se conservan y cualquier mejora posterior debe contrastarse con preguntas independientes. Los archivos de reserva utilizan el prefijo `holdout_` y no sustituyen las salidas de desarrollo.

Para reproducir por terminal la comparación y la reserva con la misma colección:

```bash
python -m src.pipeline --root . --backend local --provider e5 --retrieval dense --prefix dense_comparison
python -m src.pipeline --root . --backend local --provider e5 --retrieval rerank
python -m src.pipeline --root . --backend local --provider e5 --retrieval rerank --benchmark holdout.json --prefix holdout
```

| Ruta | Contenido |
|---|---|
| `Trayectoria_UCI_ACA1.ipynb` | Desarrollo metodológico, gráficos, ejecución y referencias |
| `src/pipeline.py` | Canalización modular y adaptadores |
| `data/raw/student_outcomes.csv` | CSV entregado, con integridad verificada |
| `data/documents/` | Destino local de los tres PDF descargados |
| `src/prepare_data.py` | Descarga y verificación SHA-256 de las fuentes |
| `data/dataset.json` | Procedencia y huella del CSV |
| `data/sources.json` | Metadatos y referencias del corpus |
| `data/evaluation.json` | Doce preguntas de desarrollo y evidencia pertinente |
| `data/holdout.json` | Seis preguntas de reserva y evidencia pertinente |
| `results/cohorts.sqlite` | Registros y perfiles agregados en SQL |
| `results/summary.json` | Configuración, conteos y métricas de la ejecución local |
| `results/cloud_summary.json` | Verificación independiente del clúster remoto y métricas de desarrollo |
| `results/cloud_search_results.json` | Respuestas reales de las doce consultas al clúster remoto |
| `results/cloud_evaluation.csv` | Evaluación de desarrollo del CLI remoto |
| `results/search_results.json` | Recuperaciones completas y latencias |
| `results/evaluation.csv` | Resultados de desarrollo por pregunta |
| `results/baseline/` | Ejecución densa histórica sin filtro de fuente, conservada por separado |
| `results/dense_comparison_summary.json` | Comparación densa reproducida durante el notebook |
| `results/holdout_summary.json` | Métricas de las seis preguntas de reserva |
| `results/holdout_search_results.json` | Evidencia completa de la reserva |
| `results/out_of_scope.json` | Control de una consulta ajena al dominio |

Conserve las atribuciones al reutilizar los materiales. El dataset UCI se distribuye bajo CC BY 4.0; los documentos institucionales mantienen sus autorías y versiones. El puntaje semántico no equivale a probabilidad de verdad ni sustituye la lectura de la fuente.


## Diagnóstico de los resultados

En el banco de desarrollo, la recuperación densa local acierta 7 de 12 preguntas y la configuración con reordenamiento acierta 9 de 12; la mejora observada añade coste de inferencia, pues la latencia mediana pasa de 24,0 a 1.519,1 ms. La reserva interna registra 5 aciertos en 6 preguntas con la configuración fijada. Estas cifras corresponden a las consultas y al entorno documentados, sin estimar el desempeño en otras poblaciones de preguntas.

El examen de los veinte candidatos localiza los fallos restantes: en T05 y T08, el pasaje esperado no entra al conjunto candidato; en T02, sí aparece en la posición densa 9, pero ocupa la posición 5 después del reordenamiento y queda fuera del Top-3. La única consulta fallida de reserva, H01, tampoco incluye la evidencia esperada entre sus veinte candidatos. La cobertura de evidencia en el conjunto candidato es 10/12 en desarrollo y 5/6 en reserva; todas las anclas existen en los fragmentos del corpus. El diagnóstico conserva preguntas, etiquetas y parámetros, y distingue la etapa en que se pierde la evidencia sin atribuirle una causa lingüística definitiva.

[El análisis por pregunta](results/error_analysis.json) conserva fuente, página, anclas, UUID, posiciones, puntuaciones, hashes de entrada y el resultado de las nueve pruebas automatizadas. Cualquier modificación posterior del recuperador debe evaluarse con preguntas nuevas, preservando esta reserva.
