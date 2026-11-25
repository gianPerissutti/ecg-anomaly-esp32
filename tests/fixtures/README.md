# Tramos de ECG para la demo

Los dos archivos contienen extractos de la derivación MLII de la base MIT-BIH Arrhythmia Database v1.0.0. Cada archivo guarda 930 muestras consecutivas a 360 Hz. WFDB lee el primer canal en mV y esas mismas muestras se codifican como `float32` little-endian para enviarlas por UART. El filtro que ubica los picos y la normalización para el modelo se aplican en el firmware.

| Archivo | Registro | Intervalo de muestras, base cero |
| --- | --- | --- |
| `mit-bih-116-251.f32le` | 116 | `[251, 1181)` |
| `mit-bih-119-193.f32le` | 119 | `[193, 1123)` |

Datos de George Moody y Roger Mark (2005). [MIT-BIH Arrhythmia Database v1.0.0, PhysioNet](https://doi.org/10.13026/C2F305) · [Open Data Commons Attribution License v1.0](https://opendatacommons.org/licenses/by/1-0/)

## Cómo cotejar los archivos con PhysioNet

Desde la raíz del repo, ejecutá este código con WFDB instalado. Lee los mismos intervalos de la base pública y comprueba que los bytes coincidan con los del repo.

```python
from pathlib import Path
from struct import pack

import wfdb

for record, start in (("116", 251), ("119", 193)):
    signal = wfdb.rdrecord(
        record, pn_dir="mitdb/1.0.0", sampfrom=start, sampto=start + 930, channels=[0]
    )
    sample_bytes = pack("<930f", *signal.p_signal[:, 0])
    fixture = Path(f"tests/fixtures/mit-bih-{record}-{start}.f32le")
    assert fixture.read_bytes() == sample_bytes, fixture
    print(f"{fixture.name} coincide con PhysioNet")
```
