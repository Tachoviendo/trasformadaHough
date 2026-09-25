# Conteo de colonias con Hough Circle Transform

Implementación del paper *"A Comparison of Bacterial Colonies Count from Petri Dishes
Utilizing Hough Transform and Traditional Manual Counting"*
([arXiv:2505.20365](https://arxiv.org/abs/2505.20365)).

Pipeline (`colony_counter.py`): detección y recorte de la placa → 512×512 (PIL) → grises +
CLAHE → sharpening Laplaciano → umbral adaptativo → Hough Circle Transform (OpenCV) →
conteo, imágenes anotadas, anotaciones YOLO y COCO.

![Ejemplo: original | binaria | colonias detectadas](demo/annotated/IMG_7710ecoli_T4_10%5E-7_59.jpg)

*E. coli, dilución 10⁻⁷: 61 colonias detectadas vs 59 contadas a mano.*

## Reproducir el experimento

### 1. Requisitos

- Python 3.12
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- ~4.5 GB libres en disco (zip + imágenes descomprimidas)

### 2. Clonar e instalar dependencias

```bash
git clone git@github.com:Tachoviendo/trasformadaHough.git
cd trasformadaHough
uv sync
```

### 3. Descargar el dataset

Es el dataset que usa el paper (ref. 31): Rodrigues, Luís y Tavaria (2022),
*Petri dishes digital images dataset of E. coli, S. aureus and P. aeruginosa*,
[figshare 10.6084/m9.figshare.20109377.v2](https://doi.org/10.6084/m9.figshare.20109377.v2),
licencia CC BY 4.0. Son 1252 fotos JPG (2.2 GB) y no está incluido en el repo.

```bash
mkdir -p data
curl -L -o data/Petri_plates.zip https://ndownloader.figshare.com/files/35973995
unzip -q data/Petri_plates.zip -d data
ls data/Petri_plates | wc -l    # 1252
```

El conteo manual viene en el nombre de cada archivo:
`IMG_7710ecoli_T4_10^-7_59.JPG` → *E. coli*, dilución 10⁻⁷, **59 colonias**.

### 4. Probar con una placa

```bash
uv run python colony_counter.py "data/Petri_plates/IMG_7710ecoli_T4_10^-7_59.JPG" -o demo
```

Genera `demo/annotated/IMG_7710ecoli_T4_10^-7_59.jpg` con tres paneles: original, binaria
y círculos detectados con el conteo (equivalente a la Fig. 3 del paper).

### 5. Correr sobre todo el dataset

```bash
uv run python colony_counter.py data/Petri_plates -o results    # ~3 min en CPU
```

- `results/counts.csv`: conteo automático y tiempo por imagen
- `results/annotated/`: panel original | binaria | círculos para cada placa
- `results/yolo/`, `results/coco.json`: bounding boxes de las colonias

Con `--no-images` no escribe los paneles anotados (más rápido).

### 6. Comparar con el conteo manual

```bash
uv run python evaluate.py --max-count 200
```

Usa solo placas con 0–200 colonias, el mismo rango que la Tabla 1 del paper. Genera:

- `results/metrics.csv`: accuracy, precision, recall, F1 y Pearson por especie
- `results/comparison.csv`: conteo manual vs automático por placa
- `results/fig4_counts_error_time.png`: conteos, error y tiempo por placa (Fig. 4)
- `results/fig5_histograms.png`: histogramas de conteos (Fig. 5)
- `results/fig6_pearson.png`: correlación manual vs automático (Fig. 6)

`evaluate.py` toma el conteo manual del nombre del archivo o de un CSV
`image,manual_count` con `--manual`. Excluye las placas con `300` (incontables) y las
marcadas "contar de novo" / "estragadas".

## Parámetros

`--param2`, `--min-dist`, `--min-radius`, `--max-radius`, `--block`, `--c`, `--rim`.
El paper no los reporta: los defaults se eligieron con un barrido sobre ~100 placas.
`uv run python colony_counter.py --help` muestra todas las opciones.

## Diferencias con el paper

- **Recorte de la placa antes de escalar**: si la foto (3:2) se escala directo a 512×512,
  la placa queda ovalada y el borde se cuenta como colonias.
- **Métricas a nivel conteo**: el dataset trae solo el total manual, no la posición de cada
  colonia, así que TP = mín(auto, manual), FP = exceso, FN = faltante.

## Resultados (placas con 0–200 colonias, n = 751)

| Especie       | Accuracy | Precision | Recall | F1   | Pearson r |
|---------------|----------|-----------|--------|------|-----------|
| E. coli       | 0.80     | 0.95      | 0.84   | 0.89 | 0.87      |
| S. aureus     | 0.73     | 0.91      | 0.78   | 0.84 | 0.73      |
| P. aeruginosa | 0.84     | 0.98      | 0.86   | 0.92 | 0.98      |

~0.14 s por placa en CPU. Funciona muy bien con pocas colonias y empeora en placas muy
llenas (colonias superpuestas). Las colonias grandes de *S. aureus* a veces reciben dos
círculos (ver `demo/annotated/IMG_8261saureus_dplaca_10^-6_96.jpg`: 131 vs 96).
