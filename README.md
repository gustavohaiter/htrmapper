# HTRMapper

Software de fotogrametria aérea para agricultura de precisão, com
tratamento correto de PPK/RTK como observações ponderadas no ajuste (não
como GPS "flag"). Ver `ARCHITECTURE.md` para a análise completa de
arquitetura, bibliotecas, riscos e o plano de fases.

Foco atual do usuário: DJI Mavic 3M, imagens já geotaggeadas via PPK,
SIRGAS 2000 / UTM 23S (EPSG:31983), agricultura sem GCP.

## Status: Fase 1 + 2 + 3 + 4 (importação → SfM → ajuste GNSS → nuvem densa)

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
- **Fase 2** (features + matching + SfM inicial, sobre `pycolmap`/COLMAP):
  extração de features SIFT, matching restrito por posição GNSS (com
  fallback exaustivo quando há poucas posições), reconstrução incremental
  e georreferenciamento por alinhamento de similaridade às coordenadas
  GNSS. CLI: `htrmapper align`. GUI: botão "Alinhar (SfM)…". Ver
  `ARCHITECTURE.md` seção 14 para detalhes e a validação com dados
  sintéticos de verdade de campo conhecida.
- **Fase 3** (bundle adjustment ponderado por GNSS/PPK — o componente
  central do projeto): refina a reconstrução da Fase 2 usando a posição
  GNSS de cada câmera como observação com peso `1/sigma²` derivado da
  `CameraAccuracy` configurada — nunca como âncora fixa. Validado
  injetando um viés de GNSS conhecido: com sigma apertado a solução segue
  o GNSS; com sigma frouxo, a reprojeção domina e o GNSS ruim é
  descartado — a prova de que o peso é matemático, não decorativo. CLI:
  `htrmapper adjust`. GUI: botão "Ajustar (GNSS)…". Preenche as tabelas
  "Camera Locations" (RMSE X/Y/Z/XY/Total em cm) e "Camera Calibration"
  do relatório. Ver `ARCHITECTURE.md` seção 15.
- **Fase 4** (nuvem de pontos densa, sobre o pipeline denso do próprio
  COLMAP): undistort → patch-match stereo → fusão estéreo → exportação
  para **LAS** (com CRS do projeto embutido, cores reais). Níveis de
  qualidade baixa/média/alta/muito_alta. CLI: `htrmapper dense`. GUI:
  botão "Nuvem densa…". **Importante:** patch-match stereo do COLMAP
  exige GPU CUDA/HIP — sem uma, falha rápido com mensagem clara (sem
  fallback de CPU nativo). O ambiente onde este projeto foi desenvolvido
  não tem GPU, então essa etapa em si (o cálculo dela) só foi validada
  quanto ao caminho de erro — a validação numérica completa depende da
  sua RTX 3060 Ti. Ver `ARCHITECTURE.md` seção 16.

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

htrmapper align projeto.json --workdir ./work --key-point-limit 40000

htrmapper adjust projeto.json --workdir ./work_ba

htrmapper dense projeto.json --workdir ./work_dense --quality media
```

## Uso — GUI

```bash
htrmapper-gui
```

No Windows, depois da instalação inicial, dá para abrir a GUI com um único
duplo-clique em `run_gui.bat` (ou rodando `run_gui.bat` no `cmd`, dentro da
pasta do projeto) — ele ativa o `.venv` automaticamente antes de chamar
`htrmapper-gui`.

## Testes

Toda a suíte usa dados sintéticos (JPEGs com EXIF + XMP DJI fabricados em
tempo de execução) — nenhum dado real de drone é necessário para rodar os
testes:

```bash
pytest -q
```

## Próximas fases

Ver `ARCHITECTURE.md`, seção "Priorização do desenvolvimento": Fase 5
(DEM), Fase 6 (ortomosaico), Fase 7 (interface completa), Fase 8
(validação quantitativa contra o Metashape), Fase 9 (otimização de
performance/GPU).
