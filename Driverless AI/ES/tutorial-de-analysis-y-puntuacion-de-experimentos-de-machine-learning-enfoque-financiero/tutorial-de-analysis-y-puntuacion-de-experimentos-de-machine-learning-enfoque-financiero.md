# Machine Learning Experiment Scoring and Analysis Tutorial - Financial Focus

## Outline
- [Objective](#objective)
- [Prerequisites](#prerequisites)
- [Task 1:  Launch Experiment](#task-1-launch-experiment)
- [Task 2: Explore Experiment Settings and Expert Settings](#task-2-explore-experiment-settings-and-expert-settings)
- [Task 3: Experiment Scoring and Analysis Concepts](#task-3-experiment-scoring-and-analysis-concepts)
- [Task 4: Experiment Results Summary and Diagnostics](#task-4-experiment-results-summary)
- [Task 5: Diagnostics Scores and Confusion Matrix](#task-5-diagnostics-scores-and-confusion-matrix)
- [Task 6: ER: ROC](#task-6-er-roc)
- [Task 7: ER: Prec-Recall](#task-7-er-prec-recall)
- [Task 8: ER: Gains](#task-8-er-gains)
- [Task 9: ER: LIFT](#task-9-er-lift)
- [Task 10: Kolmogorov-Smirnov Chart](#task-10-kolmogorov-smirnov-chart)
- [Task 11: Experiment AutoDocs](#task-11-experiment-autodocs)
- [Next Steps](#next-steps)


## Objetivo

Muchas herramientas, como ROC y curvas de recuperación de precisión, están disponibles para evaluar qué tan bueno o malo es un modelo de clasificación para predecir resultados. En este tutorial, usaremos un subconjunto del conjunto de datos de nivel de préstamo unifamiliar de Freddie Mac para construir un modelo de clasificación y lo usaremos para predecir si un préstamo quedará en mora. A través de la herramienta de diagnóstico de H2O’s Driverless AI , exploraremos los impactos financieros que tienen las predicciones de falsos positivos y falsos negativos mientras exploramos herramientas como ROC Curve, Prec-Recall, Gain and Lift Charts, K-S Chart. Finalmente, exploraremos algunas métricas como AUC, F-Scores, GINI, MCC y Log Loss para ayudarnos a evaluar el desempeño del modelo generado.

**Note:** Le recomendamos que primero revise todo el tutorial para revisar todos los conceptos, de esa manera, una vez que comience el experimento, estará más familiarizado con el contenido.
  
## Prerrequisitos
Necesitará lo siguiente para poder hacer este tutorial:

- Conocimientos básicos de aprendizaje automático y estadística.
- Un entorno de Driverless AI
- Conocimientos básicos de IA sin conductor o hacer el [Automatic Machine Learning Introduction with Drivereless AI Test Drive](https://h2oai.github.io/tutorials/automatic-ml-intro-test-drive-tutorial/#0) 

- Una **sesión de prueba de dos horas**: la prueba de manejo es Driverless AI de H2O.ai en la nube de AWS. No es necesario descargar software. Explore todas las características y beneficios de la plataforma de aprendizaje automático H2O.

  - ¿Necesita una **sesión de prueba de dos horas**? Siga las instrucciones en [este tutorial rápido](https://h2oai.github.io/tutorials/getting-started-with-driverless-ai-test-drive/#1) para iniciar una sesión de prueba de manejo.

**Note:  El laboratorio Aquarium’s Driverless AI Test Drive tiene una clave de licencia incorporada, por lo que no necesita solicitar una para usarla. Cada Driverless AI Test Drive la instancia estará disponible para usted durante dos horas, después de lo cual terminará. No se guardará ningún trabajo. Si necesita más tiempo para explorar aún más Driverless AI, siempre puede iniciar otra instancia de Test Drive o comunicarse con nuestro equipo de ventas a través del [contáctenos formulario](https://www.h2o.ai/company/contact/).**


## Tarea 1: experimento de lanzamiento

### Sobre el conjunto de datos

Este conjunto de datos contiene información sobre "datos de rendimiento crediticio a nivel de préstamo sobre una porción de hipotecas de tasa fija totalmente amortizadoras que Freddie Mac compró entre 1999 y 2017. Las características incluyen factores demográficos, rendimiento crediticio mensual, rendimiento crediticio incluyendo disposición de propiedades, pagos anticipados voluntarios, MI Recuperaciones, recuperaciones no MI, gastos, UPB diferido actual y fecha de vencimiento de la última cuota pagada."[1]

[1] Nuestro conjunto de datos es un subconjunto de [Freddie Mac Single-Family Loan-Level Dataset. ](http://www.freddiemac.com/research/datasets/sf_loanlevel_dataset.html) Contiene 500,000 filas y tiene aproximadamente 80 MB.

El subconjunto del conjunto de datos que utiliza este tutorial tiene un total de 27 características (columnas) y 500,137 préstamos (filas).

### Descargar el conjunto de datos

Descargue el subconjunto H2O del conjunto de datos de nivel de préstamo unifamiliar Freddie Mac (Freddie Mac Single-Family Loan-Level dataset) en su unidad local y guárdelo como archivo csv.

- [loan_level_500k.csv](https://s3.amazonaws.com/data.h2o.ai/DAI-Tutorials/loan_level_500k.csv)

### Lanzar experimento

1\. Carga el loan_level.csv a Driverless AI haciendo clic en **Add Dataset (agregar conjunto de datos) (or Drag and Drop (o arrastrar y soltar))** sobre el **Datasets overview (Resumen de conjuntos de datos)** página. Haga clic en **Upload File (Subir archivo)**, luego seleccione **loan_level.csv** archivo. Una vez que se carga el archivo, seleccione **Details (Detalles)**.

![loan-level-details-selection](assets/loan-level-details-selection.jpg)

**Note:** Verá cuatro conjuntos de datos más, pero puede ignorarlos, ya que trabajaremos con el`loan_level_500k.csv` archivo. 

2\. Echemos un vistazo rápido a las columnas:

![loan-level-details-page](assets/loan-level-details-page.jpg)
*Cosas a tener en cuenta:*
- C1 - CREDIT_SCORE (PUNTUACIÓN DE CRÉDITO)
- C2 - FIRST_PAYMENT_DATE (PRIMERA FECHA DE PAGO)
- C3 - FIRST_TIME_HOMEBUYER_FLAG (BANDERA DE COMPRADOR DE CASA POR PRIMERA VEZ)
- C4 - MATURITY_DATE (FECHA DE VENCIMIENTO)
- C5 - METROPOLITAN_STATISTICAL_AREA (ÁREA ESTADÍSTICA METROPOLITANA)
- C6 - MORTGAGE_INSURANCE_PERCENTAGE (PORCENTAJE DE SEGURO HIPOTECARIO)
- C7 - NUMBER_OF_UNITS (NÚMERO DE UNIDADES)

3\. Continúe desplazándose por la página actual para ver más columnas (la imagen no está incluida)
- C8 - OCCUPANCY_STATUS (ESTADO DE OCUPACIÓN)
- C9 - ORIGINAL_COMBINED_LOAN_TO_VALUE (PRÉSTAMO COMBINADO ORIGINAL AL VALOR)
- C10 - ORIGINAL_DEBT_TO_INCOME_RATIO (DEUDA ORIGINAL A LA RELACIÓN DE INGRESOS)
- C11 - ORIGINAL_UPB (ORIGINAL_UPB)
- C12 - ORIGINAL_LOAN_TO_VALUE (PRÉSTAMO ORIGINAL AL VALOR)
- C13 - ORIGINAL_INTEREST_RATE (TASA DE INTERÉS ORIGINAL)
- C14 - CHANNEL (CANAL)
- C15 - PREPAYMENT_PENALTY_MORTGAGE_FLAG (PAGO DE PREPAGO BANDERA HIPOTECARIA)
- C16 - PRODUCT_TYPE (TIPO DE PRODUCTO)
- C17- PROPERTY_STATE (Estado de la propiedad)
- C18 - PROPERTY_TYPE (TIPO DE PROPIEDAD)
- C19 - POSTAL_CODE (CÓDIGO POSTAL)
- C20 - LOAN_SEQUENCE_NUMBER (NÚMERO DE SECUENCIA DE PRÉSTAMO)
- C21 - LOAN_PURPOSE** (PROPÓSITO DEL PRESTAMO)
- C22 - ORIGINAL_LOAN_TERM (PLAZO DE PRÉSTAMO ORIGINAL)
- C23 - NUMBER_OF_BORROWERS (NÚMERO DE PRESTATARIOS)
- C24 - SELLER_NAME (NOMBRE DEL VENDEDOR)
- C25 - SERVICER_NAME (NOMBRE DEL SERVIDOR)
- C26 - PREPAID Drop (PREPAGO Drop) 
- C27 - DELINQUENT (DELINCUENTE)- Esta columna es la etiqueta que nos interesa predecir dónde Falso -> no predeterminado y Verdadero -> predeterminado


4\. Regrese a la página de resumen **Datasets**

5\. Haga clic en el **loan_level_500k.csv** archivo luego dividir (split)

![loan-level-split-1](assets/loan-level-split-1.jpg)

6\.  Divide los datos en dos conjuntos:**freddie_mac_500_train** y **freddie_mac_500_test**. Use la imagen a continuación como guía:

![loan-level-split-2](assets/loan-level-split-2.jpg)
*Cosas a tener en cuenta:*

1. Tipo ```freddie_mac_500_train``` para OUTPUT NAME 1, esto servirá como conjunto de entrenamiento
2. Tipo ```freddie_mac_500_test``` para OUTPUT NAME 2, esto servirá como conjunto de prueba
3. Para la columna de destino, seleccione **Delinquent (Delincuente)**
4. Puede establecer la semilla (Seed) aleatoria en cualquier número que desee, elegimos 42, al elegir una semilla (Seed) aleatoria obtendremos una división consistente
5. Cambie el valor de división a .75 ajustando el control deslizante a 75% o ingresando .75 en la sección que diceTrain/Valid Split Ratio (Tren / Relación de división válida)
6. Salvar


El conjunto de capacitación contiene 375k filas, cada fila representa un préstamo y 27 columnas que representan los atributos de cada préstamo, incluida la columna que tiene la etiqueta que estamos tratando de predecir.

 **Nota:** Los datos reales en la división de entrenamiento y prueba varían según el usuario, ya que los datos se dividen aleatoriamente. El conjunto de prueba contiene 125k filas, cada fila representa un préstamo y 27 columnas de atributos que representan los atributos de cada préstamo.

7\. Verifique que hay tres conjuntos de datos, **freddie_mac_500_test**, **freddie_mac_500_train** y **loan_level_500k.csv**:

![loan-level-three-datasets](assets/loan-level-three-datasets.jpg)

8\. Haga clic en el **freddie_mac_500_train** luego seleccione **Predict (Predecir)**.

9\. Seleccione **Not Now (Ahora no)** sobre el **First time Driverless AI, Haga clic en Sí para obtener un recorrido!**. Debería aparecer una imagen similar:

![loan-level-predict](assets/loan-level-predict.jpg)

Nombra tu experimento `Freddie Mac Classification Tutorial`

10\. Seleccione **Dropped Cols**, suelte las siguientes 2 columnas: 

- Prepayment_Penalty_Mortgage_Flag 
- PREPAID
- Seleccione **Done (Hecho)**

Estas dos columnas se descartan porque ambas son indicadores claros de que los préstamos se volverán morosos y causarán fugas de datos.

![train-set-drop-columns](assets/train-set-drop-columns.jpg)

 11\. Seleccione **Target Column (Columna de destino)**, luego seleccione **Delinquent**
![train-set-select-delinquent](assets/train-set-select-delinquent.jpg)

12\. Seleccione **Test Dataset (Conjunto de datos de prueba)**, luego **freddie_mac_500_test**

![add-test-set](assets/add-test-set.jpg)

13\. Debería aparecer una página de Experimento similar:   

![experiment-settings-1](assets/experiment-settings-1.jpg)    

En la tarea 2, exploraremos y actualizaremos el **Experiment Settings (Configuraciones de experimento)**.

## Tarea 2: Explorar la configuración del experimento y la configuración de expertos

1\.  Pase el mouse sobre **Experiment Settings (Configuraciones de experimento)** y tenga en cuenta las tres perillas, **Accuracy (Exactitud)**, **Time (Hora)** y **Interpretability (Interpretabilidad)**.

El **Experiment Settings** describe la precisión, el tiempo y la interpretabilidad de su experimento específico. Las perillas en la configuración del experimento son ajustables, ya que los valores cambian el significado de la configuración en la página inferior izquierda.

Aquí hay una descripción general de la configuración de Experimentos: 

- **Accuracy** - Precisión relativa: valores más altos deberían conducir a una mayor confianza en el rendimiento del modelo (precisión).
- **Time** - Tiempo relativo para completar el experimento. Los valores más altos tardarán más en completarse.
- **Interpretability**-  La capacidad de explicar o presentar en términos comprensibles a un humano. Cuanto mayor sea la interpretabilidad, más simples serán las características que se extraerán.  


### Accuracy

Al aumentar la configuración de precisión, Driverless AI ajusta gradualmente el método para realizar la evolución y el conjunto. Un conjunto de aprendizaje automático consta de múltiples algoritmos de aprendizaje para obtener un mejor rendimiento predictivo que se podría obtener de cualquier algoritmo de aprendizaje [1]. Con una configuración de baja precisión, Driverless AI varía las características (desde la ingeniería de características) y los modelos, pero todos compiten de manera uniforme entre sí. Con mayor precisión, cada modelo principal independiente evolucionará de forma independiente y será parte del conjunto final como un conjunto sobre diferentes modelos principales. Con precisiones más altas, Driverless AI evolucionará + tipos de características de conjunto, como la codificación de destino, dentro y fuera, que evolucionan de forma independiente. Finalmente, con las precisiones más altas, lDriverless AI realiza el seguimiento tanto del modelo como de las características y combina todas esas variaciones.

### Time

El tiempo especifica el tiempo relativo para completar el experimento (es decir, las configuraciones más altas tardan más). La detención temprana tendrá lugar si el experimento no mejora la puntuación para la cantidad especificada de iteraciones. Cuanto mayor sea el valor de tiempo, más tiempo se asignará para nuevas iteraciones, lo que significa que la receta tendrá más tiempo para investigar nuevas transformaciones en la ingeniería de características y el ajuste de hiperparámetros del modelo.

### Interpretability 

El mando de interpretabilidad es ajustable. Cuanto mayor sea la capacidad de interpretación, más simples serán las características que la rutina de modelado principal extraerá del conjunto de datos. Si la capacidad de interpretación es lo suficientemente alta, se generará un modelo con restricciones monotónicas.

2\.  Para este tutorial, actualice la siguiente configuración del experimento para que coincida con la imagen a continuación:
- Accuracy : 4
- Time: 3
- Interpretability: 4
- Scorer (Goleador): Logloss 

Esta configuración se seleccionó para generar un modelo rápidamente con un nivel de precisión suficiente en el entorno H2O Driverless Test Drive.

![experiment-settings-2](assets/experiment-settings-2.jpg)    

### Expert Settings (Configuraciones de expertos)

3\. Pase el mouse sobre **Expert Settings (Configuraciones de expertos)** y haga clic en él. Aparecerá una imagen similar a la siguiente:

![expert-settings-1](assets/expert-settings-1.jpg)
*Cosas a tener en cuenta:*
1. **Upload Custom Recipe (Subir receta personalizada)**
2. **Load Custom Recipe From URL (Cargar receta personalizada desde URL)** 
3. **Official Recipes (External) (Recetas oficiales (externas))**
4. **Experiment (Experimentar)**
5. **Model (Modelo)**
6. **Features (Caracteristicas)**
7. **Timeseries (Series de tiempo)**
8. **NLP**
9. **Recipes (Recetas)**
10. **System (Sistema)**

**Expert Settings** son opciones que están disponibles para aquellos que desean establecer su configuración manualmente. Explore la configuración experta disponible haciendo clic en las pestañas en la parte superior de la página.

**La configuración experta incluye (La configuración experta incluye)**:

**Configuraciones de experimento (Configuraciones de experimento)**
- Tiempo de ejecución máximo en minutos antes de activar el botón Finalizar
- Tiempo de ejecución máximo en minutos antes de activar el botón 'Abort'
- Receta de construcción de tuberías
- Hacer una tubería de puntuación de Python
- Hacer tubería de puntuación MOJO
- Medir la latencia de puntuación MOJO
- Tiempo de espera en segundos para esperar la creación de MOJO al final del experimento
- Número de trabajadores paralelos a utilizar durante la creación de MOJO
- Hacer visualización de canalización
- Hacer informe automático
- Número mínimo de filas necesarias para ejecutar un experimento
- Nivel de reproducibilidad
- Random Seed (Semilla aleatoria)
- Permitir diferentes conjuntos de clases en todos los trenes / Validation Fold Splits
- Número máximo de clases para problemas de clasificación
- Modelo / Característica Nivel cerebral
- Característica Brain Save Every, que iteración
- Característica de reinicio cerebral desde el cual iteración
- Característica Brain Reit utiliza el mismo mejor individuo
- Feature Brain agrega características con nuevas columnas incluso durante la reentrenamiento del modelo final
- Mínimas iteraciones de Driverless AI
- Seleccione la transformación de destino del objetivo para problemas de regresión
- Modelo de torneo para algoritmo genético
- Número de pliegues de validación cruzada para la evolución de características
- Número de pliegues de validación cruzada para el modelo final
- Habilite el registro adicional para Ensemble Meta Learner
- Número de pliegues de validación cruzada o divisiones máximas basadas en el tiempo para la evolución de características
- Número de pliegues de validación cruzada o divisiones máximas basadas en el tiempo para el modelo final
- Número máximo de ID de plegado para mostrar en los registros
- Número máximo de filas veces Número de columnas para divisiones de datos de evolución de características
- Número máximo de filas veces Número de columnas para reducir el conjunto de datos de entrenamiento
- Tamaño máximo de los datos de validación relativos a los datos de entrenamiento
- Realice un muestreo estratificado para la clasificación binaria si el objetivo está más desequilibrado que esto
- Agregar a config.toml a través de la cadena toml


**Configuraciones de modelo**
- Modelos XGBoost GBM
- Modelos de dardos XGBoost
- Modelos GLM
- Modelos de árbol de decisión
- Modelos LightGBM
- Modelos TensorFlow
- Modelos FTRL
- Modelos RuleFit
- Tipos de refuerzo LightGBM
- Soporte categórico LightGBM
- Modelos constantes
- Si mostrar modelos constantes en el panel de iteración
- Parámetros para TensorFlow
- Número máximo de árboles / iteraciones
- N_estimators List To Sample From para modelos que no utilizaron la detención temprana
- Tasa de aprendizaje mínima para modelos GBM de conjunto final
- Tasa máxima de aprendizaje para modelos GBM de conjunto final
- Factor de reducción para máx. Número de árboles / iteraciones durante la evolución de la característica
- Factor de reducción para el número de árboles / iteraciones durante la evolución de la característica
- Tasa de aprendizaje mínima para modelos GBM de ingeniería de características
- Tasa máxima de aprendizaje para modelos de árbol
- Número máximo de épocas para TensorFlow / FTRL
- Max. Profundidad del árbol
- Max. max_bin para las características del árbol
- Número máximo de reglas para RuleFit
- Nivel de conjunto para la tubería de modelado final
- Validación cruzada del modelo final único
- Número de modelos durante la fase de ajuste
- Método de muestreo para problemas de clasificación binaria desequilibrada
- Relación de la mayoría a la clase minoritaria para la clasificación binaria desequilibrada a las técnicas de muestreo especiales de activación (si está habilitado)
- Relación de la mayoría a la clase minoritaria para la clasificación binaria muy desequilibrada para habilitar solo técnicas de muestreo especiales si está habilitada
- Número de bolsas para métodos de muestreo para clasificación binaria desequilibrada (si está habilitada)
- Límite estricto en el número de bolsas para métodos de muestreo para clasificación binaria desequilibrada
- Límite estricto en el número de bolsas para los métodos de muestreo para la clasificación binaria desequilibrada durante la fase de evolución de características
- Tamaño máximo de datos muestreados durante el muestreo desequilibrado
- Fracción objetivo de la clase minoritaria después de aplicar técnicas de submuestreo / sobremuestreo
- Número máximo de términos de interacción automática FTRL para términos de interacción de segundo, tercer y cuarto orden (cada uno)
- Habilitar información detallada del modelo puntuado
- Si se debe habilitar el muestreo Bootstrap para la validación y los puntajes de prueba
- Para problemas de clasificación con tantas clases, el valor predeterminado es TensorFlow

**Configuración de características**
- Esfuerzo de ingeniería de características
- Detección de cambio de distribución de datos
- Distribución de datos Detección de cambio Caída de características
- Cambio de característica máximo permitido (AUC) antes de descartar la función
- Detección de fugas
- Detección de fugas que reduce el umbral de AUC / R2
- Columnas Max Rows Times para fugas
- Informe la importancia de la permutación en las características originales
- Número máximo de filas para realizar la selección de características basadas en permutación
- Número máximo de características originales utilizadas
- Número máximo de características no numéricas originales
- Número máximo de características originales utilizadas para FS Individual
- Número de características numéricas originales para activar el tipo de modelo de selección de características
- Número de características no numéricas originales para activar el tipo de modelo de selección de características
- Fracción máxima permitida de uniques para columnas enteras y categóricas
- Permitir el tratamiento numérico como categórico
- Número máximo de valores únicos para Int / Float para ser categóricos
- Número máximo de características de ingeniería
- Max. Numero de genes
- Limitar características por interpretabilidad
- Correlación más allá de la cual desencadena restricciones de monotonicidad (si está habilitada)
- Profundidad máxima de interacción de características
- Profundidad de interacción de característica fija
- Habilitar la codificación de destino
- Habilitar codificación de etiqueta lexicográfica
- Habilitar codificación de puntuación de anomalía de bosque de aislamiento
- Habilitar One HotEncoding
- Número de estimadores para la codificación de bosque de aislamiento
- Caída de columnas constantes
- Columnas de ID de caída
- No suelte ninguna columna
- Características para soltar
- Características para agrupar por
- Muestra de características para agrupar por
- Funciones de agregación (no series temporales) para agrupar por operaciones
- Número de pliegues para obtener la agregación al agrupar
- Tipo de estrategia de mutación
- Habilitar información detallada de características puntuadas
- Habilite registros detallados para el tiempo y los tipos de características producidas
- Matriz de correlación computacional

**Time Series Settings (Configuración de series de tiempo)**
- Receta basada en el retraso de la serie temporal
- Divisiones de validación personalizadas para experimentos de series temporales
- Tiempo de espera en segundos para la detección de propiedades de series temporales en la interfaz de usuario
- Generar características de vacaciones
- Anulación de retrasos de series temporales
- El tamaño de retraso más pequeño considerado
- Habilitar ingeniería de características desde la columna de tiempo
- Permitir columna de tiempo entero como función numérica
- Transformaciones de fecha y hora permitidas
- Activar ingeniería de características desde la columna de tiempo entero
- Permitir que las características de fecha u hora se transformen directamente en una representación numérica
- Considere las columnas de grupos de tiempo como características independientes
- Qué tipos de características de TGC se deben considerar como características independientes
- Habilitar transformadores de tiempo inconsciente
- Agrupar siempre por columnas de grupos de todos los tiempos para crear características de retraso
- Generar predicciones de resistencia de series temporales
- Número de divisiones basadas en el tiempo para la validación interna del modelo
- Maximum Overlap Between Two Time-Based Splits
- Número máximo de divisiones utilizadas para crear predicciones de resistencia del modelo de serie temporal final
- Si se debe acelerar el cálculo de las predicciones de resistencia de series temporales
- Si se debe acelerar el cálculo de los valores de Shapley para las predicciones de retención de series temporales
- Genere valores Shapley para predicciones de resistencia de series temporales en el momento del experimento
- Límite inferior en la configuración de interpretabilidad para experimentos de series de tiempo, aplicados implícitamente
- Modo de abandono para las características de retraso
- Probabilidad de crear características de retraso no objetivo
- Método para crear predicciones de conjuntos de pruebas continuas
- Probabilidad de que los nuevos transformadores de series temporales utilicen retrasos predeterminados
- Probabilidad de explorar transformadores de retardo basados en interacción
- Probabilidad de explorar transformadores de retardo basados en agregación

**Configuraciones de PNL**
- Max TensorFlow Epochs para PNL
- La precisión anterior habilita TensorFlow NLP de forma predeterminada para todos los modelos
- Habilitar modelos de TensorFlow CNN basados en palabras para PNL
- Habilitar modelos de TensorFlow BiGRU basados en palabras para PNL
- Habilitar modelos CNN TensorFlow basados en caracteres para PNL
- Camino a las incrustaciones preformadas para los modelos TensorFlow NLP
- Permitir el entrenamiento de incrustaciones pre-entrenadas no congeladas
- Si Python / MOJO Scoring Runtime tendrá GPU
- La fracción de columnas de texto de todas las características se considera un problema dominado por texto
- Fracción de texto por todos los transformadores para activar ese texto dominado
- Umbral para columnas de cadena que se tratarán como texto

**Configuraciones de recetas**
- Incluir transformadores específicos
- Incluir modelos específicos
- Incluir anotadores específicos
- Probabilidad de agregar transformadores
- Probabilidad de agregar los mejores transformadores compartidos
- Probabilidad de podar transformadores
- Probabilidad de mutar los parámetros del modelo
- Probabilidad de podar características débiles
- Tiempo de espera en minutos para probar la aceptación de cada receta
- Ya sea para omitir fallas de transformadores
- Ya sea para omitir fallas de modelos
- Nivel para iniciar sesión por fallas omitidas

**Ajustes del sistema**
- Número de núcleos a usar
- Número máximo de núcleos a usar para el ajuste del modelo
- Número máximo de núcleos a utilizar para predecir el modelo
- Número máximo de núcleos a usar para la transformación y predicción del modelo al hacer MLI, informe automático, puntaje en otro conjunto de datos
- Tuning Workers por lote para CPU
- Num. Funciona para entrenamiento de CPU
- #GPU/Experimento
- Num. Núcleos / GPU
- #GPU/Modelo
- Num. De GPU para predicción / transformación aislada
- Número máximo de subprocesos para usar para datatable y OpenBLAS para Munging y Model Training
- Max. Num. De subprocesos para usar en la tabla de datos de lectura y escritura de archivos
- Max. Num. De subprocesos para usar en estadísticas de tablas de datos y Openblas
- ID de inicio de GPU
- Habilitar rastros detallados
- Habilitar el nivel de registro de depuración
- Habilite el registro de la información del sistema para cada experimento


4\. Para este experimento enciéndelo **RuleFit models (Modelos RuleFit)**, debajo **Model (Modelo)** pestaña la selección **Save (Guardar)**. 

El RuleFit[2] algoritmo crea un conjunto óptimo de reglas de decisión ajustando primero un modelo de árbol y luego ajustando un modelo GLM Lasso (regularizado por L1) para crear un modelo lineal que consta de las hojas (reglas) de árbol más importantes. El modelo RuleFit ayuda a superar la precisión de los bosques aleatorios al tiempo que conserva la explicabilidad de los árboles de decisión.

![expert-settings-rulefit-on](assets/expert-settings-rulefit-on.jpg)

La activación del modelo RuleFit se agregará a la lista de algoritmos que la IA sin controlador considerará para el experimento. La selección del algoritmo depende de los datos y la configuración seleccionada.

5\. Antes de seleccionar **Launch (Lanzamiento)**, asegúrese de que su **Experiment (Experimentar)** la página es similar a la de arriba, una vez que esté lista, haga clic en **Launch (Lanzamiento)**. 

Obtenga más información sobre lo que significa cada configuración y cómo se puede actualizar a partir de sus valores predeterminados visitando la documentación de H2O- [Expert Settings](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/expert-settings.html?highlight=expert%20settings)

### Recursos

[1] [Ensemble Learning](https://en.wikipedia.org/wiki/Ensemble_learning)

[2] [J. Friedman, B. Popescu. “Predictive Learning via Rule Ensembles”. 2005](http://statweb.stanford.edu/~jhf/ftp/RuleFit.pdf)


### Inmersión más profunda 
- Para comprender mejor el impacto de establecer las perillas de precisión, tiempo e interpretabilidad entre 1 y 10 en AI sin controlador H2O](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/experiment-settings.html?highlight=interpretability#accuracy-time-and-interpretability-knobs)

- Para obtener más información sobre la configuración adicional en [Expert Settings for H2O Driverless AI](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/expert-settings.html?highlight=expert%20settings)

## Task 3: Experiment Scoring and Analysis Concepts

As we learned in the [Automatic Machine Learning Introduction Tutorial Concepts](https://github.com/h2oai/tutorials/blob/master/DriverlessAI/automatic-ml-intro-tutorial/automatic-ml-intro-tutorial.md#model-training) it is essential that once a model has been generated that its performance is evaluated. These metrics are used to evaluate the quality of the model that was built and what model score threshold should be used to make predictions  There are multiple metrics for assessing a binary classification machine learning models such as Receiver Operating Characteristics or ROC curve, Precision and Recall or Prec-Recall, Lift, Gain and K-S Charts to name a few. Each metric evaluates different aspects of the machine learning model. The concepts below are for metrics used in H2O’s Driverless AI to assess the performance of classification models that it generated. The concepts are covered at a very high level, to learn more in-depth about each metric covered here we have included additional resources at the end of this task. 


### Binary Classifier

Let’s take a look at binary classification model. A binary classification model predicts in what two categories(classes) the elements of a given set belong to. In the case of our example, the two categories(classes) are **defaulting** on your home loan and **not defaulting**. The generated model should be able to predict in which category each customer falls under.

![binary-output](assets/binary-output.jpg)

However, two other possible outcomes need to be considered, the false negative and false positives. These are the cases that the model predicted that someone did not default on their bank loan and did. The other case is when the model predicted that someone defaulted on their mortgage, but in reality, they did not. The total outcomes are visualized through a confusion matrix, which is the  two by two table seen below:

Binary classifications produce four outcomes: 

**Predicticted as Positive**:
True Positive = TP
False Positive = FP

**Predicted as Negative**:
True Negative = TN 
False Negative = FN 

![binary-classifier-four-outcomes](assets/binary-classifier-four-outcomes.jpg)

**Confusion Matrix**:

![confusion-matrix](assets/confusion-matrix.jpg)

From this confusion table, we can measure error-rate, accuracy, specificity, sensitivity, and precision, all useful metrics to test how good our model is at classifying or predicting. These metrics will be defined and explained in the next sections.

On a fun side note, you might be wondering why the name "Confusion Matrix"? Some might say that it's because a confusion matrix can be very confusing. Jokes aside, the confusion matrix is also known as the **error matrix** since it makes it easy to visualize the classification rate of the model including the error rate. The term "confusion matrix" is also used in psychology and the Oxford dictionary defines it as "A matrix representing the relative frequencies with which **each of a number of stimuli is mistaken for each of the others** by a person in a task requiring recognition or identification of stimuli. Analysis of these data allows a researcher to extract factors (2) indicating the underlying dimensions of similarity in the perception of the respondent. For example, in colour-identification tasks, relatively frequent **confusion** of reds with greens would tend to suggest daltonism." [1] In other words, how frequently does a person performing a classification task confuse one item for another. In the case of ML, a machine learning model is implementing the classification and evaluating the frequency in which the model confuses one label from another rather than a human. 

### ROC

An essential tool for classification problems is the ROC Curve or Receiver Operating Characteristics Curve. The ROC Curve visually shows the performance of a binary classifier; in other words, it  “tells how much a model is capable of distinguishing between classes” [2] and the corresponding threshold. Continuing with the Freddie Mac example the output variable or the label is whether or not the customer will default on their loan and at what threshold. 

Once the model has been built and trained using the training dataset, it gets passed through a classification method (Logistic Regression, Naive Bayes Classifier, support vector machines, decision trees, random forest, etc…), this will give the probability of each customer defaulting. 

The ROC curve plots the Sensitivity or true positive rate (y-axis) versus 1-Specificity or false positive rate (x-axis) for every possible classification threshold. A classification threshold or decision threshold is the probability value that the model will use to determine where a class belongs to. The threshold acts as a boundary between classes to determine one class from another. Since we are dealing with probabilities of values between 0 and 1 an example of a threshold can be 0.5. This tells the model that anything below 0.5 is part of one class and anything above 0.5 belongs to a different class. The threshold can be selected to maximize the true positives while minimizing false positives. A threshold is dependent on the scenario that the ROC curve is being applied to and the type of output we look to maximize. Learn more about the application of  threshold and its implications on [Task 6: ER: ROC](#task-6-er-roc).


Given our example of use case of predicting loans the following provides a description for the values in the confusion matrix:

 - TP = 1 = Prediction matches result that someone did default on a loan
 - TN = 0 = Prediction matches result that someone did not default on a loan
 - FP = 1 = Predicting that someone will default but in actuality they did not default
 - FN = 0 = Predicting that someone did not default on their bank loan but actually did.


What are sensitivity and specificity? The true positive rate is the ratio of the number of true positive predictions divided by all positive actuals. This ratio is also known as **recall** or **sensitivity**, and it is measured from 0.0 to 1.0 where 0 is the worst and 1.0 is the best sensitivity. Sensitive is a measure of how well the model is predicting for the positive case.

The true negative rate is the ratio of the number of true negative predictions divided by all positive predictions. This ratio is also known as **specificity** and is measured from 0.0 to 1.0 where 0 is the worst and 1.0 is the best specificity. Specificity is a measure for how well the model is predicting for the negative case correctly.  How often is it predicting a negative case correctly.

The false negative rate is *1- Specificity*, or it is the ratio of false positives divided by all negative predictions[3]. 

The following image provides an illustration of the ratios for sensitivity, specificity and false negative rate. 

![sensitivity-and-specificity](assets/sensitivity-and-specificity.jpg)

**Recall** = **Sensitivity** = True Positive Rate = TP / (TP + FN)

**Specificity** = True Negative Rate = TN / (FP + TN)

![false-positive-rate](assets/false-positive-rate.jpg)

**1 -Specificity** =  False Positive Rate = 1- True Negative Rate = FP / (FP + TN )

A ROC Curve is also able to tell you how well your model did by quantifying its performance. The scoring is determined by the percent of the area that is under the ROC curve otherwise known as Area Under the Curve or AUC. 

Below are four types of ROC Curves with its AUC:

**Note:** The closer the ROC Curve is to the left ( the bigger the AUC percentage), the better the model is at separating between classes. 

The Perfect ROC Curve (in red) below can separate classes with 100% accuracy and has an AUC of 1.0  (in blue):

![roc-auc-1](assets/roc-auc-1.jpg)  			

The ROC Curve below is very close to the left corner, and therefore it does a good job in separating classes with an AUC of 0.7 or 70%:

![roc-auc-07](assets/roc-auc-07.jpg)

In the case above 70% of the cases the model correctly predicted the positive and negative outcome and 30% of the cases it did some mix of FP or FN.

This ROC Curve lies on the diagonal line that splits the graph in half. Since it is further away from the left corner, it does a very poor job at distinguishing between classes, this is the worst case scenario, and it has an AUC of .05 or 50%:

![roc-auc-05](assets/roc-auc-05.jpg)

An AUC of 0.5, tells us that our model is as good as a random model that has a 50% chance of predicting the outcome. Our model is not better than flipping a coin, 50% of the time the model can correctly predict the outcome. 

Finally, the ROC Curve below represents another perfect scenario! When the ROC curve lies below the 50% model or the random chance model, then the model needs to be reviewed carefully. The reason for this is that there could have been potential mislabeling of the negatives and positives which caused the values to be reversed and hence the ROC curve is below the random chance model. Although this ROC Curve looks like it has an AUC of 0.0 or 0% when we flip it we get an AUC of 1 or 100%.

![roc-auc-0](assets/roc-auc-0.jpg)

A ROC curve is a useful tool because it only focuses on how well the model was able to distinguish between classes. “AUC’s can help represent the probability that the classifier will rank a randomly selected positive observation higher than a randomly selected negative observation” [4]. However, for models where the prediction happens rarely a high AUC could provide a false sense that the model is correctly predicting the results.  This is where the notion of precision and recall become important.

### Prec-Recall

The Precision-Recall Curve or Prec-Recall or **P-R** is another tool for evaluating classification models that is derived from the confusion matrix. Prec-Recall is a complementary tool to ROC curves, especially when the dataset has a significant skew. The Prec-Recall curve plots the precision or positive predictive value (y-axis) versus sensitivity or true positive rate (x-axis) for every possible classification threshold. At a high level, we can think of precision as a measure of exactness or quality of the results while recall as a measure of completeness or quantity of the results obtained by the model. Prec-Recall measures the relevance of the results obtained by the model.

**Precision** is the ratio of correct positive predictions divided by the total number of positive predictions. This ratio is also known as **positive predictive value** and is measured from 0.0 to 1.0, where 0.0 is the worst and 1.0 is the best precision. Precision is more focused on the positive class than in the negative class, it actually measures the probability of correct detection of positive values (TP and FP). 
 
**Precision** = True positive predictions / Total number of positive predictions = TP  / (TP + FP)

As mentioned in the ROC section, **Recall** is the true positive rate which is the ratio of the number of true positive predictions divided by all positive actuals. Recall is a metric of the actual positive predictions. It tells us how many correct positive results occurred from all the positive samples available during the test of the model.

**Recall** = **Sensitivity** = True Positive Rate = TP / (TP + FN)

![precision-recall](assets/precision-recall.jpg)

Below is another way of visualizing Precision and Recall, this image was borrowed from [https://commons.wikimedia.org/wiki/File:Precisionrecall.svg](https://commons.wikimedia.org/wiki/File:Precisionrecall.svg).

![prec-recall-visual](assets/prec-recall-visual.jpg)

A Prec-Recall Curve is created by connecting all precision-recall points through non-linear interpolation [5]. The Pre-Recall plot is broken down into two sections, “Good” and “Poor” performance. “Good” performance can be found on the upper right corner of the plot and “Poor” performance on the lower left corner, see the image below to view the perfect Pre-Recall plot. This division is generated by the baseline. The baseline for Prec-Recall is determined by the ratio of Positives(P) and Negatives(N), where y = P/(P+N), this function represents a classifier with a random performance level[6]. When the dataset is balanced, the value of the baseline is y = 0.5. If the dataset is imbalanced where the number of P’s is higher than N’s then the baseline will be adjusted accordingly and vice versa.

The Perfect Prec-Recall Curve is a combination of two straight lines (in red). The plot tells us that the model made no prediction errors! In other words, no false positives (perfect precision) and no false negatives (perfect recall) assuming a baseline of 0.5. 

![prec-recall-1](assets/prec-recall-1.jpg)

Similarly to the ROC curve, we can use the area under the curve or AUC to help us compare the performance of the model with other models. 

**Note:** The closer the Prec-Recall Curve is to the upper-right corner (the bigger the AUC percentage) the better the model is at correctly predicting the true positives. 

This Prec-Recall Curve in red below has an AUC of approximately 0.7 (in blue) with a relative baseline of 0.5:

![prec-recall-07](assets/prec-recall-07.jpg)

Finally, this Prec-Recall Curve represents the worst case scenario where the model is generating 100% false positives and false negatives. This Prec-Recall Curve has an AUC of 0.0 or 0%:

![prec-recall-00](assets/prec-recall-00.jpg)

From the Prec-Recall plot some metrics are derived that can be helpful in assessing the model’s performance, such as accuracy and Fᵦ scores.These metrics will be explained in more depth in the next section of the concepts. Just note that accuracy or ACC is the ratio number of correct predictions divided by the total number of predictions and Fᵦ is the harmonic mean of recall and precision.

When looking at ACC in Prec-Recall precision is the positive observations imperative to note that ACC does not perform well-imbalanced datasets. This is why the **F-scores** can be used to account for the skewed dataset in Prec-Recall. 

As you consider the accuracy of a model for the positive cases you want to know a couple of things:

- How often is it correct?
- When is it wrong? Why?
- Is it because you have too many false positives? (Precision)
- Or is it because you have too many false negatives?  (Recall)

There are also various  Fᵦ scores that can be considered, F1, F2 and F0.5.  The 1, 2 and 0.5 are the weights given to recall and precision. F1 for instance  means that both precision and recall have equal weight, while F2 gives recall higher weight than precision and F0.5 gives precision higher weight than recall.

Prec-Recall is a good tool to consider for classifiers because it is a great alternative for large skews in the class distribution. Use precision and recall to focus on small positive class — When the positive class is smaller and the ability to detect correctly positive samples is our main focus (correct detection of negatives examples is less important to the problem) we should use precision and recall.

If you are using a model metric of Accuracy and you see issues with Prec-Recall then you might consider using a model metric of logloss.

### GINI, ACC, F1 F0.5, F2, MCC and Log Loss

ROC and Pre-Recall curves are extremely useful to test a binary classifier because they provide visualization for every possible classification threshold. From those plots we can derive single model metrics like ACC, F1, F0.5, F2 and MCC. There are also other single metrics that can be used concurrently to evaluate models such as GINI and Log Loss. The following will be a discussion about the model scores  ACC, F1, F0.5, F2, MCC, GINI and Log Loss. The model scores are what the ML model optimizes to.

#### GINI

The Gini index is a well-established method to quantify the inequality among values of frequency distribution and can be used to measure the quality of a binary classifier. A Gini index of zero expresses perfect equality (or a totally useless classifier), while a Gini index of one expresses maximal inequality (or a perfect classifier).

The Gini index is based on the Lorenz curve. The Lorenz curve plots the true positive rate (y-axis) as a function of percentiles of the population (x-axis).

The Lorenz curve represents a collective of models represented by the classifier. The location on the curve is given by the probability threshold of a particular model. (i.e., Lower probability thresholds for classification typically lead to more true positives, but also to more false positives.)[12]

The Gini index itself is independent of the model and only depends on the Lorenz curve determined by the distribution of the scores (or probabilities) obtained from the classifier.

#### Accuracy

Accuracy or  ACC (not to be confused with AUC or area under the curve) is a single metric in binary classification problems. ACC is the ratio number of correct predictions divided by the total number of predictions. In other words, how well can the model correctly identify both the true positives and true negatives. Accuracy is measured in the range of 0 to 1, where 1 is perfect accuracy or perfect classification, and 0 is poor accuracy or poor classification[8].   

Using the confusion matrix table, ACC can be calculated in the following manner:

**Accuracy** = (TP + TN) / (TP + TN + FP + FN)

#### F-Score: F1, F0.5 and F2

The F1 Score is another measurement of classification accuracy. It represents the harmonic average of the precision and recall. F1 is measured in the range of 0 to 1, where 0 means that there are no true positives, and 1 when there is neither false negatives nor false positives or perfect precision and recall[9].

Using the confusion matrix table, the F1 score can be calculated in the following manner:

**F1** = 2TP /( 2TP + FN + FP)

**F05** equation:
F0.5 = 1.25((precision)(recall)/ 0.25precision + recall)

Where:
Precision is the positive observations (true positives) the model correctly identified from all the observations it labeled as positive (the true positives + the false positives). Recall is the positive observations (true positives) the model correctly identified from all the actual positive cases (the true positives + the false negatives)[15].

The **F2 score** is the weighted harmonic mean of the precision and recall (given a threshold value). Unlike the F1 score, which gives equal weight to precision and recall, the F2 score gives more weight to recall than to precision. More weight should be given to recall for cases where False Negatives are considered worse than False Positives. For example, if your use case is to predict which customers will churn, you may consider False Negatives worse than False Positives. In this case, you want your predictions to capture all of the customers that will churn. Some of these customers may not be at risk for churning, but the extra attention they receive is not harmful. More importantly, no customers actually at risk of churning have been missed[15].


#### MCC

MCC or Matthews Correlation Coefficient which is used as a measure of the quality of binary classifications [1]. The MCC is the correlation coefficient between the observed and predicted binary classifications. MCC is measured in the range between -1 and +1 where +1 is the perfect prediction, 0 no better than a random prediction and -1 all incorrect predictions[9].

Using the confusion matrix table MCC can be calculated in the following manner:

**MCC** =  (TP * TN- FP* FN) / [(TP + FP) * (FN + TN) * (FP + TN) * (TP + FN)] ^ ½

#### Log Loss (Logloss)
 
The logarithmic loss metric can be used to evaluate the performance of a binomial or multinomial classifier. Unlike AUC which looks at how well a model can classify a binary target, logloss evaluates how close a model’s predicted values (uncalibrated probability estimates) are to the actual target value. For example, does a model tend to assign a high predicted value like .80 for the positive class, or does it show a poor ability to recognize the positive class and assign a lower predicted value like .50? A model with a log loss of 0 would be the perfect classifier. When the model is unable to make correct predictions, the log loss increases making the model a poor model[11].

**Binary classification equation:**

![logloss-binary-classification-equation](assets/logloss-binary-classification-equation.jpg)

**Multiclass classification equation:**

![logloss-multiclass-classification-equation](assets/logloss-multiclass-classification-equation.jpg)

Where:

- N is the total number of rows (observations) of your corresponding dataframe.
- w is the per row user-defined weight (defaults is 1).
- C is the total number of classes (C=2 for binary classification).
- p is the predicted value (uncalibrated probability) assigned to a given row (observation).
- y is the actual target value.

Driverless AI Diagnostics calculates the ACC, F1, MCC values and plots those values in each ROC and Pre-Recall curves making it easier to identify the best threshold for the model generated. Additionally, it also calculates the log loss score for your model allowing you to quickly assess whether the model you generated is a good model or not. 

Let’s get back to evaluating metrics results for models.


### Gain and Lift Charts

Gain and Lift charts measure the effectiveness of a classification model by looking at the ratio between the results obtained with a trained model versus a random model(or no model)[7]. The Gain and Lift charts help us evaluate the performance of the classifier as well as answer questions such as what percentage of the dataset captured has a positive response as a function of selected percentage of a sample. Additionally, we can explore how much better we can expect do with a model compared to a random model(or no model)[7].


One way we can think of gain is “ for every step that is taken to predict an outcome the level of uncertainty decreases. A drop of uncertainty is the loss of entropy which leads to knowledge gain”[15]. The Gain Chart plots the true positive rate (sensitivity) versus the predictive positive rate(**support**) where: 

**Sensitivity** = **Recall** = True Positive Rate = TP / (TP + FN)

**Support** = **Predictive Positive Rate**  = TP + FP / (TP + FP + FN+TN) 

![sensitivity-and-support](assets/sensitivity-and-support.jpg)

To better visualize the percentage of positive responses compared to a selected percentage sample, we use **Cumulative Gains** and **Quantile**. Cumulative gains is obtained by taking the predictive model and applying it to the test dataset which is a subset of the original dataset. The predictive model will score each case with a probability. The scores are then sorted in ascending order by the predictive score. The quantile takes the total number of cases(a finite number) and partitions the finite set into subsets of nearly equal sizes. The percentile is plotted from 0th and 100th percentile. We then plot the cumulative number of cases up to each quantile starting with the positive cases  at 0%  with the highest probabilities until we reach 100% with the positive cases that scored the lowest probabilities. 

In the cumulative gains chart, the x-axis shows the percentage of cases from the total number of cases in the test dataset, while the y-axis shows the percentage of positive responses in terms of quantiles. As mentioned, since the probabilities have been ordered in ascending order we can look at the percent of predictive positive cases found in the 10% or 20% as a way to narrow down the number of positive cases that we are interested in. Visually the performance of the predictive model can be compared to that of a random model(or no model). The random model is represented below in red as the worst case scenario of random sampling.

![cumulative-gains-chart-worst-case](assets/cumulative-gains-chart-worst-case.jpg)

How can we identify the best case scenario in relation to the random model? To do this we need to identify a Base Rate first. The Base Rate sets the limits of the optimal curve. The best gains are always controlled by the Base Rate. An example of a Base Rate can be seen on the chart below (dashed green). 

- **Base Rate** is defined as:

- **Base Rate** = (TP+FN) /Sample Size

![cumulative-gains-chart-best-case](assets/cumulative-gains-chart-best-case.jpg)

The above chart represents the best case scenario of a cumulative gains chart assuming a base rate of 20%. In this scenario all the positive cases were identified before reaching the base rate.

The chart below represents an example of a predictive model (solid green curve). We can see how well the predictive model did in comparison to the random model(dotted red line). Now, we can pick a quantile and determine the percentage of positive cases up that quartile in relation to the entire test dataset. 

![cumulative-gains-chart-predictive-model](assets/cumulative-gains-chart-predictive-model.jpg)

Lift can help us answer the question of how much better one can expect to do with the predictive model compared to a random model(or no model). Lift is a measure of the effectiveness of a predictive model calculated as the ratio between the results obtained with a model and with a random model(or no model). In other words, the ratio of gain% to the random expectation % at a given quantile. The random expectation of the xth quantile is x%[16].

**Lift** = Predictive rate/ Actual rate

When plotting lift, we also plot it against quantiles in order to help us visualize how likely it is that a positive case will take place since the Lift chart is derived from the cumulative gains chart. The points of the lift curve are calculated by determining the ratio between the result predicted by our model and the result using a random model(or no model). For instance, assuming a base rate (or hypothetical threshold) of 20% from a random model, we would take the cumulative gain percent at the 20% quantile, X and divide by it by 20. We do this for all the quantiles until we get the full lift curve. 

We can start the lift chart with the base rate as seen below, recall that the base rate is the target threshold.

![lift-chart-base-rate](assets/lift-chart-base-rate.jpg)

When looking at the cumulative lift for the top quantiles, X, what it means is that when we select lets say 20% from the quantile from the total test cases based on the mode, we can expect X/20 times the total of the number of positive cases found by randomly selecting 20% from the random model.


![lift-chart](assets/lift-chart.jpg)

### K-S Chart 

Kolmogorov- Smirnov or K-S measures the performance of classification models by measuring the degree of separation between positives and negatives for validation or test data[13]. “The K-S is 100 if the scores partition the population into two separate groups in which one group contains all the positives and the other all the negatives. On the other hand, If the model cannot differentiate between positives and negatives, then it is as if the model selects cases randomly from the population. The K-S would be 0. In most classification models the K-S will fall between 0 and 100, and that the higher the value, the better the model is at separating the positive from negative cases.”[14].

The KS statistic is the maximum difference between the cumulative percentage of responders or 1's (cumulative true positive rate) and cumulative percentage of non-responders or 0's (cumulative false positive rate). The significance of KS statistic is, it helps to understand, what portion of the population should be targeted to get the highest response rate (1's)[17].

![k-s-chart](assets/k-s-chart.jpg)

### References

[1] [Confusion Matrix definition“ A Dictionary of Psychology“](http://www.oxfordreference.com/view/10.1093/acref/9780199534067.001.0001/acref-9780199534067-e-1778)

[2] [Towards Data Science - Understanding AUC- ROC Curve](https://towardsdatascience.com/understanding-auc-curve-68b2303cc9c5)

[3] [Introduction to ROC](https://classeval.wordpress.com/introduction/introduction-to-the-roc-receiver-operating-characteristics-plot/)

[4] [ROC Curves and Under the Curve (AUC) Explained](https://www.youtube.com/watch?v=OAl6eAyP-yo)

[5] [Introduction to Precision-Recall](https://classeval.wordpress.com/introduction/introduction-to-the-precision-recall-plot/)

[6] [Tharwat, Applied Computing and Informatics (2018)](https://doi.org/10.1016/j.aci.2018.08.003)

[7] [Model Evaluation Classification](https://www.saedsayad.com/model_evaluation_c.htm)

[8] [Wiki Accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision)

[9] [Wiki F1 Score](https://en.wikipedia.org/wiki/F1_score)

[10] [Wiki Matthew’s Correlation Coefficient](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient)

[11] [Wiki Log Loss](http://wiki.fast.ai/index.php/Log_Loss)

[12] [H2O’s GINI Index](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/scorers/scorers_gini.html?highlight=gini) 

[13] [H2O’s Kolmogorov-Smirnov](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/experiment-graphs.html?highlight=mcc)

[14] [Model Evaluation- Classification](https://www.saedsayad.com/model_evaluation_c.htm)

[15] [What is Information Gain in Machine Learning](https://www.quora.com/What-is-Information-gain-in-Machine-Learning)

[16] [Lift Analysis Data Scientist Secret Weapon](https://www.kdnuggets.com/2016/03/lift-analysis-data-scientist-secret-weapon.html)

[17] [Machine Learning Evaluation Metrics Classification Models](https://www.machinelearningplus.com/machine-learning/evaluation-metrics-classification-models-r/) 

### Deeper Dive and Resources

- [How and when to use ROC Curves and Precision-Recall Curves for Classification in Python](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/)

- [ROC Curves and AUC Explained](https://www.youtube.com/watch?time_continue=1&v=OAl6eAyP-yo)

- [Towards Data Science Precision vs Recall](https://towardsdatascience.com/precision-vs-recall-386cf9f89488)

- [ML Classification - Precision-Recall Curve](https://www.coursera.org/lecture/ml-classification/precision-recall-curve-rENu8)

- [Towards Data Science - Understanding and Interpreting Gain and Lift Charts](https://www.datasciencecentral.com/profiles/blogs/understanding-and-interpreting-gain-and-lift-charts)

- [ROC and AUC, Clearly Explained Video](https://www.youtube.com/watch?v=xugjARegisk)

- [What is Information gain in Machine Learning](https://www.quora.com/What-is-Information-gain-in-Machine-Learning)


## Task 4: Experiment Results Summary

At the end of the experiment, a summary of the project will appear on the right-lower corner.  Also, note that the name of the experiment is at the top-left corner.  

![experiment-results-summary](assets/experiment-results-summary.jpg)

The summary includes the following:

- **Experiment**: experiment name,
  - Version: version of Driverless AI and the date it was launched
  - Settings: selected experiment settings, seed, and amount of GPU’s enabled
  - Train data: name of the training set, number of rows and columns
  - Validation data: name of  the validation set, number of rows and columns
  - Test data: name of the test set, number of rows and columns
  - Target column: name of the target column (type of data and % target class)

- **System Specs**: machine specs including RAM, number of CPU cores and GPU’s
  - Max memory usage  

- **Recipe**: 
  - Validation scheme: type of sampling, number of internal holdouts
  - Feature Engineering: number of features scored and the final selection

- **Timing**
  - Data preparation 
  - Shift/Leakage detection
  - Model and feature tuning: total time for model and feature training and  number of models trained 
  - Feature evolution: total time for feature evolution and number of models trained 
  - Final pipeline training: total time for final pipeline training and the total models trained 
  - Python / MOJO scorer building 
- Validation Score: Log loss score +/- machine epsilon for the baseline
- Validation Score: Log loss score +/- machine epsilon for the final pipeline
- Test Score: Log loss score +/- machine epsilon score for the final pipeline 

Most of the information in the Experiment Summary tab, along with additional detail, can be found in the Experiment Summary Report (Yellow Button “Download Experiment Summary”).

Below are three questions to test your understanding of the experiment summary and frame the motivation for the following section.

1\. Find the number of features that were scored for your model and the total features that were selected. 

2\.  Take a look at the validation Score for the final pipeline and compare that value to the test score. Based on those scores would you consider this model a good or bad model?
	
**Note:** If you are not sure what Log loss is, feel free to review the concepts section of this tutorial.


3\. So what do the Log Loss values tell us?  The essential Log Loss value is the test score value. This value tells us how well the model generated did against the freddie_mac_500_test set based on the error rate. In case of experiment **Freddie Mac Classification Tutorial**, the test score LogLoss = .1180 which is the log of the misclassification rate. The greater the Log loss value the more significant the misclassification. For this experiment, the Log Loss was relatively small meaning the error rate for misclassification was not as substantial. But what would a score like this mean for an institution like Freddie Mac?

In the next few tasks we will explore the financial implications of misclassification by exploring the confusion matrix and plots derived from it. 


### Deeper Dive and Resources

- [H2O’s Experiment Summary](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/experiment-summary.html?highlight=experiment%20overview)

- [H2O’s Internal Validation](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/internal-validation.html) 


## Task 5: Diagnostics Scores and Confusion Matrix

Now we are going to run a model diagnostics on the freddie_mac_500_test set. The diagnostics model allows you to view model performance for multiple scorers based on an existing model and dataset through the Python API.

1\. Select **Diagnostics** 


![diagnostics-select](assets/diagnostics-select.jpg)

2\. Once in the **Diagnostics** page, select **+ Diagnose Model**

![diagnose-model](assets/diagnose-model.jpg)

3\. In the **Create new model diagnostics** : 
1. Click on Diagnosed Experiment then select the experiment that you completed in Task 4: **Freddie Mac Classification Tutorial**
2. Click on Dataset then select the freddie_mac_500_test dataset
3.  Initiate the diagnostics model by clicking on **Launch Diagnostics** 

![create-new-model-diagnostic](assets/create-new-model-diagnostic.jpg)

4\.After the model diagnostics is done running, a model similar to the one below will appear:

![new-model-diagnostics](assets/new-model-diagnostics.jpg) 

*Things to Note:*

1. Name of new diagnostics model
2. **Model**: Name of ML model used for diagnostics
3. **Dataset**: name of the dataset used for diagnostic
4. **Message** : Message regarding new diagnostics model 
5. **Status** : Status of new diagnostics model
6. **Time** : Time it took for the  new diagnostics model to run
7. Options for this model

5\. Click on the new diagnostics model and a page similar to the one below will appear:

![diagnostics-model-results](assets/diagnostics-model-results.jpg)

*Things to Note:*

1. **Info**: Information about the diagnostics model including the name of the test dataset, name of the experiment used and the target column used for the experiment
2. **Scores**: Summary for the values for GINI, MCC, F05, F1, F2, Accuracy, Log loss, AUC and AUCPR in relation to how well the experiment model scored against a “new” dataset

    -  **Note:** The new dataset must be the same format and with the same number of columns as the training dataset 

3. **Metric Plots**: Metrics used to score the experiment model including ROC Curve, Pre-Recall Curve, Cumulative Gains, Lift Chart, Kolmogorov-Smirnov Chart, and Confusion Matrix

4. **Download Predictions**: Download the diagnostics predictions
 
**Note:** The scores will be different for the train dataset and the validation dataset used during  the training of the model.

#### Confusion Matrix 

As mentioned in the concepts section, the confusion matrix is the root from where most metrics used to test the performance of a model originate. The confusion matrix provides an overview performance of a supervised model’s ability to classify.

Click on the confusion matrix located on the **Metrics Plot** section of the Diagnostics page, bottom-right corner. An image similar to the one below will come up:


![diagnostics-confusion-matrix-0](assets/diagnostics-confusion-matrix-0.jpg)

The confusion matrix lets you choose a desired threshold for your predictions. In this case, we will take a closer look at the confusion matrix generated by the Driverless AI model with the default threshold, which is 0.5.

The first part of the confusion matrix we are going to look at is the **Predicted labels** and **Actual labels**.  As shown on the image below the **Predicted label** values for **Predicted Condition Negative** or  **0** and **Predicted Condition Positive** or **1**  run vertically while the **Actual label** values for **Actual Condition Negative** or **0** and **Actual Condition Positive** or **1** run horizontally on the matrix.

Using this layout, we will be able to determine how well the model predicted the people that defaulted and those that did not from our Freddie Mac test dataset. Additionally, we will be able to compare it to the actual labels from the test dataset.

![diagnostics-confusion-matrix-1](assets/diagnostics-confusion-matrix-1.jpg)

Moving into the inner part of the matrix, we find the number of cases for True Negatives, False Positives, False Negatives and True Positive. The confusion matrix for this model generated tells us that :

- TP = 1 = 213 cases were predicted as **defaulting** and **defaulted** in actuality 
- TN = 0 = 120,382 cases were predicted as **not defaulting** and **did not default** 
- FP = 1 = 155 cases were predicted as **defaulting** when in actuality they **did not default**
- FN = 0 = 4,285 cases were predicted as **not defaulting** when in actuality they **defaulted**

![diagnostics-confusion-matrix-2](assets/diagnostics-confusion-matrix-2.jpg)

The next layer we will look at is the **Total** sections for **Predicted label** and **Actual label**. 

On the right side of the confusion matrix are the totals for the **Actual label**  and at the base of the confusion matrix, the totals for the **Predicted label**.

**Actual label**
- 120,537 : the number of actual cases that did not default on the test dataset
- 4,498 : the number of actual cases that defaulted on the test

**Predicted label**
- 124,667 : the number of cases that were predicted to not default on the test dataset
- 368 :  the number of cases that were predicted to default on the test dataset 

![diagnostics-confusion-matrix-3](assets/diagnostics-confusion-matrix-3.jpg)

The final layer of the confusion matrix we will explore are the errors. The errors section is one of the first places where we can check how well the model performed. The better the model does at classifying labels on the test dataset the lower the error rate will be. The **error rate** is also known as the **misclassification rate** which answers the question of how often is the model wrong?

For this particular model these are the errors:
- 155/120537 = 0.0012 or 0.12%  times the model classified actual cases that did not default as defaulting out of the actual non-defaulting group
- 4285/4498 = 0.952 or 95.2% times the model classified actual cases that did default as not defaulting out of the actual defaulting group
- 4285/124667 = 0.0343 or 3.43% times the model classified predicted cases that did default as not defaulting out of the total predicted not defaulting group
- 210/368 = 0.5706 or 57.1% times the model classified predicted cases that defaulted as defaulting out of the total predicted defaulting group
- (4285 + 155) / 125035 = **0.0355**  This means that this model incorrectly classifies  .0355 or 3.55% of the time.
 
What does the misclassification error of .0355 mean?
One of the best ways to understand the impact of this misclassification error is to look at the financial implications of the False Positives and False Negatives. As mentioned previously, the False Positives represent the loans predicted not to default and in reality did default. 
Additionally, we can look at the mortgages that Freddie Mac missed out on by not granting loans because the model predicted that they would default when in reality they did not default. 

One way to look at the financial implications for Freddie Mac is to look at the total paid interest rate per loan. The mortgages on this dataset are traditional home equity loans which means that the loans are:
- A fixed borrowed amount
- Fixed interest rate
- Loan term and monthly payments are both fixed

For this tutorial, we will assume a 6% Annual Percent Rate(APR) over 30 years. APR is the amount one pays to borrow the funds. Additionally, we are going to assume an average home loan of $167,473(this average was calculated by taking the sum of all the loans on the freddie_mac_500.csv dataset and dividing it by 30,001 which is the total number of mortgages on this dataset). For a mortgage of $167,473 the total interest paid after 30 years would be $143,739.01[1]. 

When looking at the False Positives, we can think about 155 cases of people which the model predicted should be not be granted a home loan because they were predicted to default on their mortgage. These 155 loans translate to over 18 million dollars in loss of potential income (155 * $143,739.01) in interest.

Now, looking at the True Positives, we do the same and take the 4,285 cases that were granted a loan because the model predicted that they would not default on their home loan. These 4,285 cases translate to about over 618 million dollars in interest losses since the 4,285 cases defaulted.

The misclassification rate provides a summary of the sum of the False Positives and False Negatives divided by the total cases in the test dataset. The misclassification rate for this model was .0355.  If this model were used to determine home loan approvals, the mortgage institutions would need to consider approximately 618 million dollars in losses for misclassified loans that got approved and shouldn’t have and 18 million dollars on loans that were not approved since they were classified as defaulting.

One way to look at these results is to ask the question: is missing out on approximately 18 million dollars from loans that were not approved better than losing about 618 million dollars from loans that were approved and then defaulted? There is no definite answer to this question, and the answer depends on the mortgage institution. 

![diagnostics-confusion-matrix-4](assets/diagnostics-confusion-matrix-4.jpg)

#### Scores 
Driverless AI conveniently provides a summary of the scores for the performance of the model given the test dataset.

The scores section provides a summary of the Best Scores found in the metrics plots:
- **GINI**
- **MCC**
- **F1**
- **F2**
- **Accuracy**
- **Logloss**
- **AUC**
- **AUCPR**

The image below represents the scores for the **Freddie Mac Classification Tutorial** model using the freddie_mac_500_test dataset:


![diagnostics-scores](assets/diagnostics-scores.jpg)

When the experiment was run for this classification model, Driverless AI determined that the best scorer for it was the Logarithmic Loss or **LOGLOSS** due to the imbalanced nature of the dataset. **LOGLOSS** focuses on getting the probabilities right (strongly penalizes wrong probabilities). The selection of Logarithmic Loss makes sense since we want a model that can correctly classify those who are most likely to default while ensuring that those that qualify for a loan get can get one.

Recall that Log loss is the logarithmic loss metric that can be used to evaluate the performance of a binomial or multinomial classifier, where a model with a Log loss of 0 would be the perfect classifier. Our model  scored  a LOGLOSS value = .1193+/- .0017 after testing it with test dataset. From the confusion matrix, we saw that the model had issues classifying perfectly; however, it was able to classify with an ACCURACY of .9647 +/- .0006. The financial implications of the misclassifications have been covered in the confusion matrix section above.

Driverless AI has the option to change the type of scorer used for the experiment. Recall that for this dataset the scorer was selected to be **logloss**. An experiment can be re-run with another scorer. For general imbalanced classification problems, AUCPR and MCC scorers are good choices, while F05, F1, and F2 are designed to balance recall against precision.
The AUC is designed for ranking problems. Gini is similar to the AUC but measures the quality of ranking (inequality) for regression problems. 

In the next few tasks we will explore the scorer further and the **Scores** values in relation to the residual plots.

### References

[1] [Amortization Schedule Calculator](https://investinganswers.com/calculators/loan/amortization-schedule-calculator-what-repayment-schedule-my-mortgage-2859) 

### Deeper Dive and Resources

- [Wiki Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix)

- [Simple guide to confusion matrix](https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)

- [Diagnosing a model with Driverless AI](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/diagnosing.html)

## Task 6: ER: ROC

From the Diagnostics page click on the **ROC Curve**. An image similar to the one below will appear:

![diagnostics-roc-curve](assets/diagnostics-roc-curve.jpg)

To review, an ROC curve demonstrates the following:

- It shows the tradeoff between sensitivity (True Positive Rate or TPR) and specificity (1-FPR or False Positive Rate). Any increase in sensitivity will be accompanied by a decrease in specificity.
- The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the model.
- The closer the curve comes to the 45-degree diagonal of the ROC space, the less accurate the model.
- The slope of the tangent line at a cutpoint gives the likelihood ratio (LR) for that value of the test. You can check this out on the graph above.
- The area under the curve is a measure of model accuracy. 

Going back to the Freddie Mac dataset, even though the model was scored with the Logarithmic Loss to penalize for error we can still take a look at the ROC curve results and see if it supports our conclusions from the analysis of the confusion matrix and scores section of the diagnostics page.

1\. Based on the ROC curve that Driverless AI model generated for your experiment, identify the AUC. Recall that a perfect classification model has an AUC of 1.

2\. For each of the following points on the curve, determine the True Positive Rate, False Positive rate, and threshold by hovering over each point below as seen on the image below:
- Best Accuracy 
- Best F1
- Best MCC

![diagnostics-roc-best-acc](assets/diagnostics-roc-best-acc.jpg)

Recall that for a binary classification problem, accuracy is the number of correct predictions made as a ratio of all predictions made.  Probabilities are converted to predicted classes in order to define a threshold. For this model, it was determined that the best accuracy is found at threshold .5375.

At this threshold, the model predicted:
- TP = 1 = 175 cases predicted as defaulting and defaulted
- TN = 0 = 120,441 cases predicted as not defaulting and did not default
- FP = 1 = 96 cases predicted as defaulting and did not default
- FN = 0 = 4,323 cases predicted to not default and defaulted


3\.  From the AUC, Best MCC, F1, and Accuracy values from the ROC curve, how would you qualify your model, is it a good or bad model? Use the key points below to help you asses the ROC Curve.


Remember that for the **ROC** curve: 
- The perfect classification model has an AUC of 1
- MCC is measured in the range between -1 and +1 where +1 is the perfect prediction, 0 no better than a random prediction and -1 all incorrect predictions.
- F1 is measured in the range of 0 to 1, where 0 means that there are no true positives, and 1 when there is neither false negatives nor false positives or perfect precision and recall.
- Accuracy is measured in the range of 0 to 1, where 1 is perfect accuracy or perfect classification, and 0 is poor accuracy or poor classification.

**Note:** If you are not sure what AUC, MCC, F1, and Accuracy are or how they are calculated review the concepts section of this tutorial.

### New Model with Same Parameters

In case you were curious and wanted to know if you could improve the accuracy of the model, this can be done by changing the scorer from Logloss to Accuracy.  

1\. To do this, click on the **Experiments**  page.

2\. Click on the experiment you did for task 1 and select **New Model With Same Params**

![new-model-w-same-params](assets/new-model-w-same-params.jpg)

An image similar to the one below will appear. Note that this page has the same settings as the setting in Task 1. The only difference is that on the **Scorer** section **Logloss** was updated to **Accuracy**. Everything else should remain the same.

3\. If you haven’t done so, select **Accuracy** on the scorer section then select **Launch Experiment**

![new-model-accuracy](assets/new-model-accuracy.jpg)

Similarly to the experiment in Task 1, wait for the experiment to run. After the experiment is done running, a similar page will appear. Note that on the summary located on the bottom right-side both the validation and test scores are no longer being scored by **Logloss** instead by **Accuracy**. 

![new-experiment-accuracy-summary](assets/new-experiment-accuracy-summary.jpg)

We are going to use this new experiment to run a new diagnostics test. You will need the name of the new experiment. In this case, the experiment name is **1.Freddie Mac Classification Tutorial**. 

4\. Go to the **Diagnostics** tab.

5\. Once in the **Diagnostics** page, select **+Diagnose Model**

6\. In the **Create new model diagnostics** : 
1. Click on Diagnosed Experiment then select the experiment that you completed in Task in this case the experiment name is **1.Freddie Mac Classification Tutorial** 
2. Click on Dataset then select the freddie_mac_500_test dataset
3. Initiate the diagnostics model by clicking on **Launch Diagnostics** 

![diagnostics-create-new-model-for-accuracy](assets/diagnostics-create-new-model-for-accuracy.jpg)

7\. After the model diagnostics is done running a new diagnostic will appear

8\. Click on the new diagnostics model. On the **Scores** section observe the accuracy value. Compare this Accuracy value to the Accuracy value from task 6. 

![diagnostics-scores-accuracy-model](assets/diagnostics-scores-accuracy-model.jpg)


9\. Next, locate the new ROC curve and click on it. Hover over the **Best ACC** point on the curve. An image similar to the one below will appear:


![diagnostics-roc-curve-accuracy-model](assets/diagnostics-roc-curve-accuracy-model.jpg)

How much improvement did we get from optimizing the accuracy via the scorer? 

The new model predicted:
- Threshold = .5532
- TP =  1 =  152 cases predicted as defaulting and defaulted
- TN = 0 = 120,463  cases predicted as not defaulting and did not default
- FP = 1 = 74 cases predicted as defaulting and did not default
- FN = 0 = 4,346 cases predicted not to default and defaulted

The first model predicted:
- Threshold = .5375
- TP = 1 = 175 cases predicted as defaulting and defaulted
- TN = 0 = 120,441 cases predicted as not defaulting and did not default
- FP = 1 = 96 cases predicted as defaulting and did not default
- FN = 0 = 4,323 cases predicted to not default and defaulted

The threshold for best accuracy changed from .5375 for the first diagnostics model to .5532 for the new model. This increase in threshold improved accuracy or the number of correct predictions made as a ratio of all predictions made. Note, however, that while the number of FP decreased the number of FN increased.  We were able to reduce the number of cases that were predicted to falsy default, but in doing so, we increased the number of FN or cases that were predicted not to default and did.

The takeaway is that there is no win-win; sacrifices need to be made. In the case of accuracy, we increased the number of mortgage loans, especially for those who were denied a mortgage because they were predicted to default when, in reality, they did not. However, we also increased the number of cases that should not have been granted a loan and did.  As a mortgage lender, would you prefer to reduce the number of False Positives or False Negatives?

10\. Exit out of the ROC curve by clicking on the **x** located at the top-right corner of the plot, next to the **Download** option

### Deeper Dive and Resources

- [How and when to use ROC Curves and Precision-Recall Curves for Classification in Python](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/)

- [ROC Curves and AUC Explained](https://www.youtube.com/watch?time_continue=1&v=OAl6eAyP-yo)
- [Towards Data Science - Understanding AUC- ROC Curve](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5)

- [ROC Curves and Under the Curve (AUC) Explained](https://www.youtube.com/watch?v=OAl6eAyP-yo)

- [Introduction to ROC](https://classeval.wordpress.com/introduction/introduction-to-the-roc-receiver-operating-characteristics-plot/)


## Task 7: ER: Prec-Recall

Continuing on the diagnostics page, select the **P-R** curve. The P-R curve should look similar to the one below:

![diagnostics-pr-curve](assets/diagnostics-prec-recall.jpg)

Remember that for the **Prec-Recall**:

- The precision-recall plot uses recall on the x-axis and precision on the y-axis. 
- Recall is identical to sensitivity, and precision is identical to the positive predictive value.
- ROC curves should be used when there are roughly equal numbers of observations for each class.
- Precision-Recall curves should be used when there is a moderate to large class imbalance.
- Similar to ROC, the AUCPR (Area under the curve of Precision-recall curve) is a measure of model accuracy and higher the better. 
- In both the ROC and Prec-recall curve, Driverless AI will indicate points that are the best thresholds for Accuracy (ACC), F1 or MCC (Matthews correlation coefficient).

Looking at the  P-R curve results, is this a good model to determine if a customer will default on their home loan? Let’s take a look at the values found on the P-R curve.

1\. Based on the P-R curve that Driverless AI model generated for you experiment identify the AUC.

2\. For each of the following points on the curve, determine the True Positive Rate, False Positive rate, and threshold by hovering over each point below as seen on the image below:
- Best Accuracy 
- Best F1
- Best MCC

![diagnostics-prec-recall-best-mccr](assets/diagnostics-prec-recall-best-mcc.jpg)

3\.  From the observed AUC, Best MCC, F1 and Accuracy values for P-R, how would you qualify your model, is it a good or bad model? Use the key points below to help you asses the P-R curve.

Remember that for the **P-R** curve :

- The perfect classification model has an AUC of 1
- MCC is measured in the range between -1 and +1 where +1 is the perfect prediction, 0 no better than a random prediction and -1 all incorrect predictions.
- F1 is measured in the range of 0 to 1, where 0 means that there are no true positives, and 1 when there is neither false negatives nor false positives or perfect precision and recall.
- Accuracy is measured in the range of 0 to 1, where 1 is perfect accuracy or perfect classification, and 0 is poor accuracy or poor classification.


**Note:** If you are not sure what AUC, MCC, F1, and Accuracy are or how they are calculated review the concepts section of this tutorial.

### New Model with Same Parameters

Similarly to task 6, we can improve the area under the curve for precision-recall by creating a new model with the same parameters. Note that you need to change the Scorer from **Logloss** to **AUCPR**. You can try this on your own. 

To review how to run a new experiment with the same parameters and a different scorer, follow the step on task 6, section **New Model with Same Parameters**.

![new-model-w-same-params-aucpr](assets/new-model-w-same-params-aucpr.jpg)

**Note:** If you ran the new experiment, go back to the diagnostic for the experiment we were working on.

### Deeper Dive and Resources

- [Towards Data Science Precision vs Recall](https://towardsdatascience.com/precision-vs-recall-386cf9f89488)

- [ML Classification - Precision-Recall Curve](https://www.coursera.org/lecture/ml-classification/precision-recall-curve-rENu8)

- [Introduction to Precision-Recall](https://classeval.wordpress.com/introduction/introduction-to-the-precision-recall-plot/)

## Task 8: ER: Gains

 Continuing on the diagnostics page, select the **CUMULATIVE GAIN** curve. The Gains curve should look similar to the one below:

![diagnostics-gains](assets/diagnostics-gains.jpg)

Remember that for the **Gains** curve:

- A cumulative gains chart is a visual aid for measuring model performance. 
- The y-axis shows the percentage of positive responses. This is a percentage of the total possible positive responses 
- The x-axis shows the percentage of all customers from the Freddie Mac dataset who did not default, which is a fraction of the total cases
- The dashed line is the baseline (overall response rate)
- It helps answer the question of  “What fraction of all observations of the positive target class are in the top predicted 1%, 2%, 10%, etc. (cumulative)?” By definition, the Gains at 100% are 1.0.

**Note:** The y-axis of the plot has been adjusted to represent quantiles, this allows for focus on the quantiles that have the most data and therefore the most impact.

1\. Hover over the various quantile points on the Gains chart to view the quantile percentage and cumulative gain values

2\. What is the cumulative gain at  1%, 2%, 10% quantiles?

![diagnostics-gains-10-percent](assets/diagnostics-gains-10-percent.jpg)

For this Gain Chart, if we look at the top 1% of the data, the at-chance model (the dotted diagonal line) tells us that we would have correctly identified 1% of the defaulted mortgage cases. The model generated (yellow curve) shows that it was able to identify about 12% of the defaulted mortgage cases. 

If we hover over to the top 10% of the data, the at-chance model (the dotted diagonal line) tells us that we would have correctly identified 10% of the defaulted mortgage cases. The model generated (yellow curve) says that it was able to identify about 53% of the defaulted mortgage cases. 

3\. Based on the shape of the gain curve and the baseline (white diagonal dashed line) would you consider this a good model? 

Remember that the perfect prediction model starts out pretty steep, and as a rule of thumb the steeper the curve, the higher the gain. The area between the baseline (white diagonal dashed line) and the gain curve (yellow curve) better known as the area under the curve visually shows us how much better our model is than that of the random model. There is always room for improvement. The gain curve can be steeper.

**Note:** If you are not sure what AUC or what the gain chart is, feel free to review the concepts section of this tutorial.

4\. Exit out of the Gains chart by clicking on the **x** located at the top-right corner of the plot, next to the **Download** option

### Deeper Dive and Resources
 
- [Towards Data Science - Understanding and Interpreting Gain and Lift Charts](https://www.datasciencecentral.com/profiles/blogs/understanding-and-interpreting-gain-and-lift-charts)

## Task 9: ER: LIFT

Continuing on the diagnostics page, select the **LIFT** curve. The Lift curve should look similar to the one below:

![diagnostics-lift](assets/diagnostics-lift.jpg)

Remember that for the **Lift** curve:

A Lift chart is a visual aid for measuring model performance.

- Lift is a measure of the effectiveness of a predictive model calculated as the ratio between the results obtained with and without the predictive model.
- It is calculated by determining the ratio between the result predicted by our model and the result using no model.
- The greater the area between the lift curve and the baseline, the better the model.
- It helps answer the question of “How many times more observations of the positive target class are in the top predicted 1%, 2%, 10%, etc. (cumulative) compared to selecting observations randomly?” By definition, the Lift at 100% is 1.0.

**Note:**  The y-axis of the plot has been adjusted to represent quantiles, this allows for focus on the quantiles that have the most data and therefore the most impact.


1\. Hover over the various quantile points on the Lift chart to view the quantile percentage and cumulative lift values

2\. What is the cumulative lift at 1%, 2%, 10% quantiles?
![diagnostics-lift-10-percent](assets/diagnostics-lift-10-percent.jpg)
For this Lift Chart, all the predictions were sorted according to decreasing scores generated by the model. In other words, uncertainty increases as the quantile moves to the right. At the 10% quantile, our model predicted a cumulative lift of about 5.3%, meaning that among the top 10% of the cases, there were five times more defaults.

3\. Based on the area between the lift curve and the baseline (white horizontal dashed line) is this a good model?

The area between the baseline (white horizontal dashed line) and the lift curve (yellow curve) better known as the area under the curve visually shows us how much better our model is than that of the random model. 

4\. Exit out of the Lift chart by clicking on the **x** located at the top-right corner of the plot, next to the **Download** option

### Deeper Dive and Resources

- [Towards Data Science - Understanding and Interpreting Gain and Lift Charts](https://www.datasciencecentral.com/profiles/blogs/understanding-and-interpreting-gain-and-lift-charts)


## Task 10: Kolmogorov-Smirnov Chart

Continuing on the diagnostics page, select the **KS** chart. The K-S chart should look similar to the one below:

![diagnostics-ks](assets/diagnostics-ks.jpg)

Remember that for the K-S chart:

- K-S measures the performance of classification models by measuring the degree of separation between positives and negatives for validation or test data.
- The K-S is 100 if the scores partition the population into two separate groups in which one group contains all the positives and the other all the negatives
- If the model cannot differentiate between positives and negatives, then it is as if the model selects cases randomly from the population and the K-S would be 0
- The K-S range is between 0 and 1
- The higher the K-S value, the better the model is at separating the positive from negative cases

**Note:** The y-axis of the plot has been adjusted to represent quantiles, this allows for focus on the quantiles that have the most data and therefore the most impact.

1\. Hover over the various quantile points on the Lift chart to view the quantile percentage and cumulative lift values

2\. What is the cumulative lift at 1%, 2%, 10% quantiles?


![diagnostics-ks-20-percent](assets/diagnostics-ks-20-percent.jpg)

For this K-S chart, if we look at the top  20% of the data, the at-chance model (the dotted diagonal line) tells us that only 20% of the data was successfully separate between positives and negatives (defaulted and not defaulted). However, with the model it was able to do .5508 or about 55% of the cases were successfully separated between positives and negatives.

3\. Based on the K-S curve(yellow) and the baseline (white diagonal dashed line) is this a good model?


4\. Exit out of the K-S chart by clicking on the **x** located at the top-right corner of the plot, next to the **Download** option

### Deeper Dive and Resources

- [Kolmogorov-Smirnov Test](https://towardsdatascience.com/kolmogorov-smirnov-test-84c92fb4158d)
- [Kolmogorov-Smirnov Goodness of Fit Test](https://www.statisticshowto.datasciencecentral.com/kolmogorov-smirnov-test/)


## Task 11: Experiment AutoDocs

Driverless AI makes it easy to download the results of your experiments, all at the click of a button.  

1\. Let’s explore the auto generated documents for this experiment. On the **Experiment** page select **Download Experiment Summary**.

![download-experiment-summary](assets/download-experiment-summary.jpg)

The **Experiment Summary** contains the following:

- Summary of Experiment
- Experiment Features along with relevant importance
- Ensemble information
- Experiment preview
- The auto-generated report for the experiment in .docx format
- Train data summary in a csv format
- Target transformations tuning leaderboard
- Leaderboard

A **report** file is included in the **experiment** summary. This report provides insight into the training data and any detected shifts in distribution, the validation schema selected, model parameter tuning, feature evolution and the final set of features chosen during the experiment.

2\. Open the report .docx file, this auto-generated report contains the following information:
- Experiment Overview
- Data Overview
- Methodology
- Data Sampling
- Validation Strategy
- Model Tuning
- Feature Evolution
- Feature Transformation
- Final Model
- Alternative Models
- Deployment
- Appendix

3\. Take a few minutes to explore the report

4\. Explore Feature Evolution and Feature Transformation, how is this summary different from the summary provided in the **Experiments Page**?

5\. Find the section titled **Final Model** on the report.docx and explore the following items:
- Table titled **Performance of Final Model** and determine the **logloss** final test score
- Validation Confusion Matrix
- Test Confusion Matrix
- Validation and Test ROC, Prec-Recall, lift, and gains plots

### Deeper Dive and Resources

- [H2O’s Summary Report](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/experiment-summary.html?highlight=experiment%20overview)


## Next Steps

Check out the next tutorial : [Machine Learning Interpretability](https://h2oai.github.io/tutorials/machine-learning-experiment-scoring-and-analysis-tutorial-financial-focus/#0) where you will learn how to:
- Launch an experiment
- Create ML interpretability report
- Explore explainability concepts such as:
    - Global Shapley
    - Partial Dependence plot
    - Decision tree surrogate
    - K-Lime
    - Local Shapley
    - LOCO
    - Individual conditional Expectation









