# HTRMapper

Software de fotogrametria aérea para agricultura de precisão, com
tratamento correto de PPK/RTK como observações ponderadas no ajuste (não
como GPS "flag"). Ver `ARCHITECTURE.md` para a análise completa de
arquitetura, bibliotecas, riscos e o plano de fases.

Foco atual do usuário: DJI Mavic 3M, imagens já geotaggeadas via PPK,
SIRGAS 2000 / UTM 23S (EPSG:31983), agricultura sem GCP.

## Status: Fase 1 + 2 + 3 + 4 + 5 + 6 + 7 (importação → SfM → ajuste GNSS → nuvem densa → DEM → ortomosaico → interface completa)

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
  GNSS. Imagens são reduzidas para no máximo 2000px no lado maior antes da
  extração de features (`--max-image-size`, padrão validado contra fotos
  reais de 21MP — ver `ARCHITECTURE.md` seção 20 para por que esse valor
  específico, e não um mais "redondo", é o que realmente funciona). CLI:
  `htrmapper align`. GUI: botão "Alinhar (SfM)…". Ver `ARCHITECTURE.md`
  seção 13 para detalhes e a validação com dados sintéticos de verdade de
  campo conhecida.
- **Fase 3** (bundle adjustment ponderado por GNSS/PPK — o componente
  central do projeto): refina a reconstrução da Fase 2 usando a posição
  GNSS de cada câmera como observação com peso `1/sigma²` derivado da
  `CameraAccuracy` configurada — nunca como âncora fixa. Validado
  injetando um viés de GNSS conhecido: com sigma apertado a solução segue
  o GNSS; com sigma frouxo, a reprojeção domina e o GNSS ruim é
  descartado — a prova de que o peso é matemático, não decorativo. CLI:
  `htrmapper adjust`. GUI: botão "Ajustar (GNSS)…". Preenche as tabelas
  "Camera Locations" (RMSE X/Y/Z/XY/Total em cm) e "Camera Calibration"
  do relatório. Ver `ARCHITECTURE.md` seção 14.
- **Fase 4** (nuvem de pontos densa, sobre o pipeline denso do próprio
  COLMAP): undistort → patch-match stereo → fusão estéreo → exportação
  para **LAS** (com CRS do projeto embutido, cores reais). Níveis de
  qualidade baixa/média/alta/muito_alta. CLI: `htrmapper dense`. GUI:
  botão "Nuvem densa…". **Importante:** patch-match stereo do COLMAP
  exige GPU CUDA/HIP — sem uma, falha rápido com mensagem clara (sem
  fallback de CPU nativo). O ambiente onde este projeto foi desenvolvido
  não tem GPU, então essa etapa em si (o cálculo dela) só foi validada
  quanto ao caminho de erro — a validação numérica completa depende da
  sua RTX 3060 Ti. Ver `ARCHITECTURE.md` seção 15.
- **Fase 5** (DEM/DSM a partir da nuvem densa, via GDAL/`rasterio` +
  `scipy.interpolate`): filtro de outliers robusto (mediana + MAD, não
  um corte de percentil fixo — um bug real pego pelo próprio teste, ver
  `ARCHITECTURE.md`), resolução automática (2,5x o espaçamento médio de
  pontos) ou definida pelo usuário, interpolação linear + preenchimento
  de buracos por nearest-neighbor, GeoTIFF com CRS/geotransform/NoData
  corretos. Validado contra uma superfície sintética de verdade
  conhecida: erro médio < 15cm, erro máximo < 50cm comparado ao valor
  real da função de terreno, não apenas "rodou sem erro". CLI:
  `htrmapper dem`. GUI: botão "Gerar DEM…". Ver `ARCHITECTURE.md` seção 16.
- **Fase 6** (ortomosaico — **2.5D**, não true-ortho 3D completo, ver
  `ARCHITECTURE.md`): reprojeta as imagens não-distorcidas da Fase 4 no
  plano do DEM da Fase 5, reaproveitando o próprio modelo de câmera do
  COLMAP (`cam_from_world`/`img_from_cam`), com blending por
  distância-à-borda ("feathering") entre câmeras sobrepostas. Saída
  GeoTIFF RGBA (alpha = NoData explícito onde nenhuma câmera viu o
  terreno). Validado contra cores de terreno conhecidas de uma cena
  sintética com posições de câmera exatas (erro médio de cor < 8/255) —
  processo que expôs e corrigiu um bug real de convenção norte-sul na
  fixture de teste compartilhada, sem afetar as fases anteriores. CLI:
  `htrmapper ortho`. GUI: botão "Gerar Ortomosaico…". Ver
  `ARCHITECTURE.md` seção 17.
- **Fase 7** (interface completa): árvore de projeto (Projeto / Imagens /
  Câmeras / Tie Points / Point Cloud / DEM / Orthomosaic). Cada etapa de
  longa duração (Alinhar, Ajustar, Nuvem densa, DEM, Ortomosaico) roda em
  segundo plano (`QThread`) em vez de travar a interface, com barra de
  progresso (real, nunca uma porcentagem inventada — indeterminada para
  as etapas apoiadas no COLMAP, que não expõe percentual; determinada e
  granular no Ortomosaico, que é código próprio por-câmera) e um botão
  Cancelar que interrompe de verdade via `pycolmap.CancellationToken` —
  quando a etapa suporta cancelamento nativo; a Fase 3 (ajuste GNSS) não
  oferece Cancelar porque o solver do Ceres não expõe esse gancho nesta
  versão do pycolmap, nunca um botão decorativo. Monitor de CPU/RAM/GPU/
  VRAM ao vivo (atualizado a cada segundo). Testando essas duas coisas
  juntas expôs e corrigiu um deadlock real de `fork()` (chamar
  `nvidia-smi` enquanto o COLMAP tem várias threads nativas ativas). GUI:
  `htrmapper-gui`. Ver `ARCHITECTURE.md` seção 18.

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

htrmapper dem projeto.json --output dem.tif

htrmapper ortho projeto.json --output ortho.tif
```

## Uso — GUI

```bash
htrmapper-gui
```

No Windows, dá para abrir a GUI com um único duplo-clique em `run_gui.bat`
(ou rodando `run_gui.bat` no `cmd`, dentro da pasta do projeto) — ele cria
o `.venv` se ainda não existir, ativa e roda `pip install -e ".[gui,dev]"`
sempre (não só na primeira vez, para captar automaticamente uma
dependência nova de uma fase mais recente num `.venv` criado antes dela —
ex. `pycolmap` da Fase 2), antes de chamar `htrmapper-gui`. Depois de um
`git pull`, é só rodar `run_gui.bat` de novo normalmente.

## Testes

Toda a suíte usa dados sintéticos (JPEGs com EXIF + XMP DJI fabricados em
tempo de execução) — nenhum dado real de drone é necessário para rodar os
testes. Os testes da GUI (Fase 7) rodam com o plugin de plataforma Qt
"offscreen" (`QT_QPA_PLATFORM=offscreen`, já configurado no próprio
arquivo de teste) — não é preciso um display real nem X11/Wayland:

```bash
pytest -q
```

## Próximas fases

Ver `ARCHITECTURE.md`, seção "Priorização do desenvolvimento": Fase 8
(validação quantitativa contra o Metashape), Fase 9 (otimização de
performance/GPU).
