# HTRMapper

Software de fotogrametria aérea para agricultura de precisão, com
tratamento correto de PPK/RTK como observações ponderadas no ajuste (não
como GPS "flag"). Ver `ARCHITECTURE.md` para a análise completa de
arquitetura, bibliotecas, riscos e o plano de fases.

Foco atual do usuário: DJI Mavic 3M, imagens já geotaggeadas via PPK,
SIRGAS 2000 / UTM 23S (EPSG:31983), agricultura sem GCP.

## Status: Fase 1 (importação, EXIF/XMP, CRS, visualização, config. GNSS)

Implementado nesta fase:

- Importação de pasta de imagens (JPG/TIFF).
- Leitura de EXIF (posição GPS, altitude, focal length, câmera, timestamp).
- Leitura de XMP específico da DJI (orientação do gimbal, orientação de
  voo, e metadados de RTK do próprio drone quando presentes).
- Relatório de importação: imagens sem coordenadas, coordenadas
  inválidas, extensão espacial, altitude min/max, câmeras detectadas.
- Sistema de CRS correto via PROJ/pyproj (nunca conversão manual),
  separando CRS de origem / CRS do projeto / CRS de exportação.
- Modelo de precisão GNSS/PPK (`CameraAccuracy`: sigma XY, sigma Z) e a
  matemática de peso (`1/sigma^2`) que alimentará o bundle adjustment
  ponderado por GNSS na Fase 3.
- Formato de projeto próprio (JSON versionado), salvar/carregar.
- CLI (`htrmapper import <pasta> ...`) para uso sem GUI / automação.
- Visualizador desktop mínimo (PySide6): tabela de imagens + mapa real de
  posições de câmera no CRS do projeto.
- Relatório de processamento (HTML), com a mesma estrutura de um relatório
  Metashape (Survey Data, Camera Calibration, Camera Locations, DEM,
  Orthomosaic, Processing Parameters, System) — ver `ARCHITECTURE.md`
  seção 11 para o mapeamento campo a campo. Tudo que ainda depende de
  fases futuras (tie points, bundle adjustment, nuvem densa, DEM,
  ortomosaico) aparece explicitamente como **"Não disponível — calculado
  na Fase N"**, nunca como zero ou valor inventado.

## Instalação (desenvolvimento)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[gui,dev]"
```

## Uso — CLI

```bash
htrmapper import /caminho/para/imagens \
    --epsg 31983 \
    --xy-sigma 0.02 --z-sigma 0.02 \
    --project-out projeto.json \
    --report-out relatorio.html
```

## Uso — GUI

```bash
htrmapper-gui
```

## Testes

Toda a suíte usa dados sintéticos (JPEGs com EXIF + XMP DJI fabricados em
tempo de execução) — nenhum dado real de drone é necessário para rodar os
testes:

```bash
pytest -q
```

## Próximas fases

Ver `ARCHITECTURE.md`, seção "Priorização do desenvolvimento": Fase 2
(features/matching/SfM via COLMAP), Fase 3 (bundle adjustment com pesos
GNSS via Ceres), Fase 4 (nuvem densa), Fase 5 (DEM), Fase 6 (ortomosaico),
Fase 7 (interface completa), Fase 8 (validação quantitativa contra o
Metashape), Fase 9 (otimização de performance/GPU).
