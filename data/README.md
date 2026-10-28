# Datos y documentos de TRAYECTORIA-UCI

El repositorio conserva los manifiestos de procedencia e integridad; los PDF originales se obtienen desde las publicaciones institucionales. Desde la raíz del proyecto, ejecute:

```bash
python -m src.prepare_data --root .
```

El preparador utiliza únicamente la biblioteca estándar de Python. Descarga los recursos que faltan y verifica los SHA-256 declarados en `dataset.json` y `sources.json`; los archivos existentes que coinciden se conservan sin volver a descargarlos. Si un archivo existe con otro hash, el programa lo conserva y comunica el error. Las descargas sólo se instalan tras verificar su integridad; las fallidas no reemplazan archivos.

Para comprobar una copia ya preparada sin acceder a Internet:

```bash
python -m src.prepare_data --root . --verify-only
```

Los reintentos son limitados: tres por URL de manera predeterminada, con espera de conexión de veinte segundos. Una publicación institucional puede cambiar o dejar de estar disponible; en ese caso, el preparador informa el fallo y mantiene los hashes de la versión empleada en el proyecto.

## Dataset de base

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students’ Dropout and Academic Success* [Conjunto de datos]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5MC89](https://doi.org/10.24432/C5MC89).

Licencia: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). El archivo empleado corresponde a la copia proporcionada por el usuario; el preparador puede reconstruir sus mismos bytes CSV desde el ZIP público de UCI. Se comprueba `csv_sha256`, manteniendo el hash del ZIP aportado como registro de procedencia, pues la compresión del contenedor puede cambiar sin alterar el CSV. Se conservan los valores originales y su codificación; la limpieza de encabezados y las transformaciones analíticas se realizan durante la ejecución del proyecto.

Los 4.424 registros proceden de una institución portuguesa. El corpus colombiano proporciona referencias temáticas sobre bienestar, acompañamiento y apoyos; no establece correspondencia entre personas ni permite atribuir a los registros UCI las políticas de las instituciones colombianas.

## Corpus documental

| Autor y año | Documento | Procedencia |
|---|---|---|
| Ministerio de Educación Nacional (2016) | *Lineamientos de política de bienestar para instituciones de educación superior* | [Publicación ministerial](https://www.mineducacion.gov.co/1780/articles-360314_recurso.pdf); [copia institucional en la Universidad de San Buenaventura Cali](https://usbcali.edu.co/wp-content/uploads/2023/10/lineamientos_de_politica_de_bienestar_para_ies.pdf) |
| Universidad El Bosque (2024) | *Política para el éxito estudiantil y lineamientos* | [Publicación institucional](https://www.unbosque.edu.co/sites/default/files/2024-10/PoliticadeExitoEstudiantil.pdf) |
| Universidad Nacional de Colombia, Vicerrectoría Académica, Dirección Nacional de Programas de Posgrado (2022) | *Guía de acompañamiento para estudiantes y tutores de Posgrado de la Universidad Nacional de Colombia* (4.ª ed.) | [Publicación institucional](https://posgrados.unal.edu.co/wp-content/uploads/2024/01/GuiaAcompanamientoEstudiantesTutores_Agosto2022.pdf) |

Los documentos mantienen los derechos de sus respectivos titulares. No se identificó en ellos una licencia abierta explícita; su disponibilidad pública no se interpreta como autorización general de redistribución. Por esta razón, el repositorio aporta referencias, metadatos y un procedimiento de descarga desde sus fuentes institucionales. Cualquier licencia del código del proyecto se limita a ese código y no se extiende a estos documentos.

`sources.json` registra las versiones, las páginas físicas de los PDF, las URL originales y de descarga, la autoría y los hashes. La versión ministerial se obtiene primero de la copia institucional de la Universidad de San Buenaventura Cali; la URL del MEN se conserva como procedencia y alternativa. La consulta del corpus informa sobre estas ediciones y no certifica su vigencia normativa posterior ni resuelve elegibilidad individual para apoyos.

## Bancos de comprobación

`evaluation.json` contiene el banco inicial de doce preguntas y `holdout.json`, una reserva interna de seis preguntas. Sus etiquetas remiten a documento, página física y expresión identificable del pasaje. Los archivos de preparación no modifican estos bancos ni sus hashes.
