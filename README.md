# Conteo de colonias con Hough Circle Transform

Implementación del paper *"A Comparison of Bacterial Colonies Count from Petri Dishes
Utilizing Hough Transform and Traditional Manual Counting"* (`~/Downloads/Ignacio.pdf`).

Pipeline (`colony_counter.py`): detección y recorte de la placa → 512×512 (PIL) → grises +
CLAHE → sharpening Laplaciano → umbral adaptativo → Hough Circle Transform (OpenCV) →
conteo, imágenes anotadas, anotaciones YOLO y COCO.

## Uso

```bash
uv run python colony_counter.py data/Petri_plates -o results   # o una sola imagen
uv run python evaluate.py --max-count 200                       # métricas vs conteo manual
```

- `results/counts.csv` – conteo automático y tiempo por imagen
- `results/annotated/` – original | binaria | círculos detectados (Fig. 3 del paper)
- `results/yolo/`, `results/coco.json` – bounding boxes de las colonias
- `results/metrics.csv`, `comparison.csv`, `fig4_*.png`, `fig5_*.png`, `fig6_*.png`

`evaluate.py` toma el conteo manual del nombre del archivo del dataset
(`IMG_7710ecoli_T4_10^-7_59.JPG` → 59) o de un CSV `image,manual_count` con `--manual`.
Excluye las placas con `300` (incontables) y las marcadas "contar de novo" / "estragadas".

Parámetros ajustables: `--param2`, `--min-dist`, `--min-radius`, `--max-radius`,
`--block`, `--c`, `--rim` (defaults elegidos con un barrido sobre ~100 placas).

## Dataset

Rodrigues et al. 2022, figshare 20109377 (ref. 31 del paper), en `data/Petri_plates/`.

## Resultados (placas con 0–200 colonias, n = 751)

| Especie       | Accuracy | Precision | Recall | F1   | Pearson r |
|---------------|----------|-----------|--------|------|-----------|
| E. coli       | 0.80     | 0.95      | 0.84   | 0.89 | 0.87      |
| S. aureus     | 0.73     | 0.91      | 0.78   | 0.84 | 0.73      |
| P. aeruginosa | 0.84     | 0.98      | 0.86   | 0.92 | 0.98      |

~0.14 s por placa en CPU. Métricas a nivel conteo: TP = min(auto, manual),
FP = exceso, FN = faltante.
# trasformadaHough
