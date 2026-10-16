# TP3 — Mecánica del Sólido Computacional

**Análisis de una presa de gravedad mediante el Método de los Elementos Finitos (MEF)**

Trabajo práctico N.º 3 de la materia **30.53 — Mecánica del Sólido Computacional**. Se desarrolló en Python un programa modular para resolver el equilibrio estático de un sólido elástico lineal bidimensional y aplicarlo a una presa de gravedad sometida a peso propio y presión hidrostática.

El repositorio incluye cuatro formulaciones de elementos finitos, pruebas de verificación, lectura y generación de mallas, postprocesamiento de tensiones y desplazamientos, análisis de estabilidad y estudios de convergencia y variación geométrica.

## 1. Modelo físico

Se analiza una sección trapezoidal con paramento de aguas arriba vertical y paramento de aguas abajo inclinado.

| Parámetro | Valor |
|---|---:|
| Altura de la presa | 30 m |
| Ancho de base | 18 m |
| Ancho de cresta | 3 m |
| Borde libre | 2 m |
| Nivel del agua desde la base | 28 m |
| Espesor longitudinal de referencia | 1 m |
| Módulo de Young del hormigón | 21 GPa |
| Coeficiente de Poisson | 0,20 |
| Densidad del hormigón | 2400 kg/m³ |
| Densidad del agua | 1000 kg/m³ |
| Aceleración gravitatoria | 9,81 m/s² |

**Hipótesis mecánicas:** material homogéneo, isótropo y elástico lineal; pequeñas deformaciones y equilibrio estático. Se adopta **deformación plana (`plane_state="strain"`)**, de acuerdo con la indicación del profesor y no a la consigna. En esta hipótesis, `εz = γxz = γyz = 0`, aunque `σz` puede ser distinta de cero.

**Condiciones de borde:** se prescribe `ux = uy = 0` en todos los nodos de la base (`y = 0`). Este empotramiento idealiza una fundación mucho más rígida que la estructura; no representa explícitamente deformaciones del suelo, deslizamiento ni apertura del contacto. Las restantes superficies están libres de tracciones, excepto el paramento aguas arriba sobre el que actúa el agua.

La presión hidrostática aplicada es `p(y) = ρagua · g · (28 − y)` para `0 ≤ y ≤ 28 m`, y nula por encima del nivel del agua. Se considera suelo impermeable, pero **no se modela subpresión bajo la presa**.

Las hipótesis, sus condiciones de validez y las limitaciones del modelo se desarrollan en [`hipotesis.txt`](hipotesis.txt).

## 2. Elementos y características del código

| Elemento | Geometría | Interpolación | Nodos por elemento |
|---|---|---|---:|
| CST | Triangular | Lineal | 3 |
| LST | Triangular | Cuadrática | 6 |
| Q4 | Cuadrilateral | Bilineal | 4 |
| Q8 | Cuadrilateral | Cuadrática serendípita | 8 |

El programa implementa:

- Ensamblaje de la matriz global de rigidez y resolución del sistema lineal con restricciones de desplazamiento.
- Integración de peso propio y presión hidrostática mediante fuerzas nodales equivalentes.
- Cálculo de reacciones y comprobaciones de equilibrio global de fuerzas y momentos.
- Lectura, escritura y generación de mallas estructuradas para la sección trapezoidal.
- Desplazamientos, deformaciones, tensiones normales y cortantes, y tensiones principales.
- Visualización de la malla deformada con factor de amplificación seleccionable.
- Representación de la **máxima tensión principal tridimensional sobre la configuración deformada**, incluyendo `σz` para el estado de deformación plana.
- Cálculo del momento estabilizante, el momento desestabilizante y un factor de seguridad simplificado frente al vuelco.
- Estudios paramétricos de refinamiento de malla y variaciones de la geometría.

## 3. Requisitos e instalación

Se recomienda **Python 3.12**. Las bibliotecas externas utilizadas por el código son **NumPy** y **Matplotlib**; las demás importaciones son de la biblioteca estándar de Python.

Desde la raíz del repositorio, en PowerShell (Windows):

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy matplotlib
```

En Linux o macOS:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install numpy matplotlib
```

## 4. Ejecución

Todos los programas principales están ubicados en la **raíz del proyecto** y pueden ejecutarse desde la terminal o con **▶ Run Python File** en VS Code, seleccionando el intérprete del entorno virtual.

### 4.1. Análisis principal de la presa

```powershell
python main.py
```

En `main.py` se pueden modificar:

- `element_type`: `"CST"`, `"LST"`, `"Q4"` o `"Q8"`.
- `nx`, `ny`: divisiones horizontales y verticales de la malla.
- `generate_mesh`: generar una malla nueva o utilizar un archivo existente.
- `height`, `base_width`, `crest_width`: geometría de la presa.
- `scale_factor`: amplificación **solo gráfica** de los desplazamientos.
- `plot`: habilitar o deshabilitar los gráficos (en la configuración de `DamConfig`).

La configuración incluida utiliza Q8 con una malla `6×10` (60 elementos). El programa imprime resultados y verificaciones de carga, estabilidad y equilibrio. Cuando se abren ventanas de Matplotlib, puede ser necesario cerrarlas para que continúe la ejecución.

Los gráficos principales se guardan en `output/`, por ejemplo:

- `deformedMeshpresa_Q8_6x10.png`: malla original y deformada.
- `displacementspresa_Q8_6x10.png`: componentes y módulo del desplazamiento.
- `stressespresa_Q8_6x10.png`: campos de tensiones.
- `principalStressDeformedpresa_Q8_6x10.png`: máxima tensión principal sobre la configuración deformada.

### 4.2. Estudio de convergencia

```powershell
python runConvergence.py
```

Ejecuta automáticamente CST, LST, Q4 y Q8 en cinco niveles de refinamiento (`3×5`, `6×10`, `9×15`, `12×20`, `15×25`). Compara:

- Desplazamiento horizontal en `(3, 30) m`.
- Tensión vertical en `(2, 10.3) m`.
- Energía total de deformación `Ed = 1/2 · UᵀKU`.
- Variaciones relativas entre niveles y grados de libertad totales.

Los CSV, las mallas generadas y los gráficos individuales se guardan en `results/convergence/<ELEMENTO>/`.

Para **regenerar solamente la comparación** a partir de los CSV existentes:

```powershell
python compareConvergence.py
```

Genera `convergenceComparison.csv` y los gráficos `comparisonDisplacement.png`, `comparisonStress.png` y `comparisonEnergy.png` en `results/convergence/comparison/`.

**Advertencia:** las variaciones relativas entre mallas son indicadores de estabilización, no errores respecto de una solución exacta. Las tensiones puntuales, especialmente cerca de esquinas empotradas o interfaces de elementos, son sensibles al refinamiento.

### 4.3. Estudio de variaciones geométricas

```powershell
python runGeometryStudy.py
```

Analiza combinaciones de ancho de base y ancho de cresta con mallas Q8. Compara área de hormigón, factor de seguridad simplificado al vuelco, indicadores de tracción y compresión alejados de las esquinas, y tensiones clásicas de base. También estima anchos mínimos para factores de seguridad objetivo.

Las tablas y gráficos se guardan en `results/geometry/`:

- `geometryStudy.csv` y `minimumBase.csv`.
- `geometryFS.png`, `geometryEfficiency.png` y `geometryStresses.png`.

## 5. Pruebas y verificaciones

Los siguientes archivos pueden ejecutarse individualmente desde VS Code con **Play**, o desde la raíz del repositorio:

```powershell
python tests/testCST.py
python tests/testLST.py
python tests/testQ4.py
python tests/testQ8.py
python tests/testMesh.py
python tests/testPostprocess.py
python tests/testStability.py
python tests/testGeometryStudy.py
python tests/testPrincipalStressDeformed.py
```

Se comprueban, según el test, interpolación y Jacobiano, matrices de rigidez, modos rígidos, patch tests, desplazamientos y tensiones analíticas, integridad del formato de malla, equilibrio de fuerzas y momentos, estabilidad y representación de la tensión principal deformada.

Para el caso base, las verificaciones analíticas de referencia incluyen:

| Magnitud | Valor |
|---|---:|
| Área de la sección | 315 m² |
| Peso propio (por 1 m de espesor) | 7,41636 MN |
| Resultante hidrostática | 3,84552 MN |
| Momento desestabilizante respecto del pie aguas abajo | 35,89152 MN·m |
| Momento estabilizante respecto del pie aguas abajo | 87,93684 MN·m |
| Factor de seguridad simplificado al vuelco | ≈ 2,45 |

Los momentos de las cargas FEM se contrastan con los momentos analíticos y con las reacciones obtenidas en la base.

## 6. Formato de archivo de malla

Se utiliza un formato de texto plano (`.mesh`) con tres secciones principales:

```text
MATERIALS n_materiales
# id  E[Pa]  nu  rho[kg/m3]  estado_plano
1 2.1e10 0.2 2400 strain

NODES n_nodos
# id  x[m]  y[m]
...

ELEMENTS n_elementos
# id  tipo  orden  id_material  nodos...
...
```

El lector verifica la consistencia entre el tipo de elemento, el orden de interpolación y la cantidad de nodos, además de validar la estructura del archivo. Los archivos de referencia se encuentran en `meshes/`.

## 7. Organización del repositorio

```text
MSC_Tp3/
├── main.py                    # Análisis principal
├── runConvergence.py          # Convergencia por formulación
├── compareConvergence.py      # Gráficos comparativos
├── runGeometryStudy.py        # Estudio de geometría
├── hipotesis.txt              # Hipótesis y límites del modelo
├── meshes/                   # Mallas de entrada y generadas
├── source/
│   ├── elements/             # CST, LST, Q4, Q8
│   ├── materials/            # Elasticidad lineal y hormigón
│   ├── mesh/                 # Lectura, generación y métricas
│   ├── loads/                # Peso propio y presión hidrostática
│   ├── systems/              # Ensamblaje y condiciones de borde
│   ├── solvers/              # Resolución estática
│   ├── problems/             # Modelo de la presa de gravedad
│   ├── output/               # Reportes, estabilidad y mapas
│   └── studies/              # Convergencia y geometría
├── tests/                    # Pruebas automatizadas
├── output/                   # Gráficos del análisis principal
└── results/                  # CSV y resultados de los estudios
```

## 8. Alcance y limitaciones

El programa resuelve un modelo **elástico lineal, de pequeñas deformaciones y con base perfectamente empotrada**. El factor de seguridad calculado es un cociente simplificado de momentos frente al vuelco: **no equivale a una verificación geotécnica ni estructural integral**. No se modelan subpresión, filtración, sismo, deslizamiento, apertura de contacto, fisuración ni comportamiento no lineal del hormigón.


Para una discusión más detallada de la validez de las condiciones de borde y las hipótesis físicas, consultar [`hipotesis.txt`](hipotesis.txt).
