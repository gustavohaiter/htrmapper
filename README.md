# HTRMapper

Software de fotogrametria aérea para agricultura de precisão, com
tratamento correto de PPK/RTK como observações ponderadas no ajuste (não
como GPS "flag"). Ver `ARCHITECTURE.md` para a análise completa de
arquitetura, bibliotecas, riscos e o plano de fases.

Foco atual do usuário: DJI Mavic 3M, imagens já geotaggeadas via PPK,
SIRGAS 2000 / UTM 23S (EPSG:31983), agricultura sem GCP.

## Começo rápido (Windows + placa NVIDIA)

**PC novo:** copie esta pasta para o PC (ou baixe o ZIP do GitHub), entre em
`windows\` e dê duplo clique em `INSTALAR.bat`. Ele confere o Windows e a
placa NVIDIA, instala o WSL e o Ubuntu (pode pedir para reiniciar e rodar de
novo), instala o HTRMapper com a GPU dentro do Ubuntu e cria o atalho
**HTRMapper** na Área de Trabalho. Detalhes em `ARCHITECTURE.md` seção 31.

**PC que já roda o HTRMapper no WSL** (terminal do Ubuntu):

```bash
cd ~/htrmapper && git pull
bash windows/instalar_wsl.sh --criar-atalhos
```

Depois é só abrir pelo atalho **HTRMapper**: escolher a pasta das fotos, a
pasta de saída e o nível (Baixo / Médio / Alto), e clicar em **Processar**.
Ele faz alinhamento, ajuste com o RTK, nuvem densa, DEM, ortomosaico e o
relatório PDF sozinho, mostrando cada etapa e o tempo restante. Se der erro,
mostra na tela e salva um `ERRO_<data>.txt` na pasta de saída; rodando de
novo, continua de onde parou. Para atualizar: Menu Iniciar → HTRMapper →
**Atualizar HTRMapper**.

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
  específico, e não um mais "redondo", é o que realmente funciona), e cada
  imagem é comparada contra no máximo 30 vizinhas espaciais no matching
  (`--spatial-max-neighbors` — o corte pós-matching que ferramentas como o
  Metashape chamam de "Tie Point Limit" foi pesquisado e testado, mas
  comprovadamente não afeta o tempo de matching; ver `ARCHITECTURE.md`
  seções 21-22, que também documentam por que a Fase 4 não usa GPU mesmo
  em máquinas com placa NVIDIA real). CLI:
  `htrmapper align`. GUI: botão "Alinhar (SfM)…". Ver `ARCHITECTURE.md`
  seção 13 para detalhes e a validação com dados sintéticos de verdade de
  campo conhecida.
- **Fase 3** (bundle adjustment ponderado por GNSS/PPK — o componente
  central do projeto): refina a reconstrução da Fase 2 usando a posição
  GNSS de cada câmera como observação com peso `1/sigma²` derivado da
  `CameraAccuracy` configurada — nunca como âncora fixa. Validado
  injetando um viés de GNSS conhecido: com sigma apertado a solução segue
  o GNSS; com sigma frouxo, a reprojeção domina e o GNSS ruim é
  descartado — a prova de que o peso é matemático, não decorativo. A lente
  (focal, ponto principal, distorção) é refinada junto, com travas contra
  valores implausíveis, e cada foto pesa pelo próprio desvio RTK quando
  ele é maior que o configurado. No voo real isso levou o erro das
  câmeras RTK FIX de ~30 cm para ~2 cm (ver `ARCHITECTURE.md` seção 27).
  CLI: `htrmapper adjust` (`--fix-intrinsics` mantém a lente fixa). GUI: botão "Ajustar (GNSS)…". Preenche as tabelas
  "Camera Locations" (RMSE X/Y/Z/XY/Total em cm) e "Camera Calibration"
  do relatório. Ver `ARCHITECTURE.md` seção 14.
- **Fase 4** (nuvem de pontos densa, sobre o pipeline denso do próprio
  COLMAP): undistort → patch-match stereo → fusão estéreo → exportação
  para **LAS** (com CRS do projeto embutido, cores reais). Níveis de
  qualidade baixa/média/alta/muito_alta. CLI: `htrmapper dense`. GUI:
  botão "Nuvem densa…". **Importante:** patch-match stereo do COLMAP
  exige GPU CUDA/HIP — sem uma, falha rápido com mensagem clara (sem
  fallback de CPU nativo). **Validado com hardware real em 2026-10-06/07**
  (RTX 3060 Ti via WSL2, 56 fotos reais, qualidade "alta": 90.246 pontos
  fundidos) — essa execução também achou e corrigiu um bug real na
  chamada do `pycolmap.stereo_fusion`, e expôs uma densidade de pontos
  inesperadamente baixa (~0,87 pontos/m²). Causa raiz provável: o MVS do
  COLMAP trabalha em `float32`, que em coordenadas UTM absolutas (Y ≈ 7,5
  milhões) só resolve passos de 0,5 m — agora a nuvem densa roda num
  referencial local e a origem é somada de volta na exportação.
  Confirmado no mesmo voo real: 2,58 milhões de pontos contra 158 mil,
  no mesmo tempo; ver `ARCHITECTURE.md` seções 15, 24 e 25.
- **Fase 5** (DEM/DSM a partir da nuvem densa, via GDAL/`rasterio` +
  `scipy`): filtro global de erros grosseiros (mediana + MAD), mediana
  dos pontos por célula, filtro de picos local (como o "noise filtering"
  do Pix4D, com proteção por componente conectado para não apagar
  valetas, carreadores ou terraços), buracos preenchidos só onde a
  superfície foi reconstruída em volta (como o modo padrão do Metashape)
  e NoData fora dela, grade alinhada a múltiplos da resolução, GeoTIFF
  em blocos com compressão. Resolução automática de 2x o espaçamento
  médio dos pontos ou definida pelo usuário. Num banco de testes com
  superfície conhecida (talhão com linhas de cana, valeta, buraco e
  pontos espúrios): erro RMS de 10 para 5 cm, picos acima de 30 cm de
  1,25% para 0,08% das células, sem os 4.600 m² inventados fora da área
  com dados, 20x mais rápido. CLI: `htrmapper dem`. GUI: botão "Gerar
  DEM…". Ver `ARCHITECTURE.md` seções 16 e 28.
- **Fase 6** (ortomosaico — **2.5D**, não true-ortho 3D completo, ver
  `ARCHITECTURE.md`): reprojeta as imagens não-distorcidas da Fase 4 no
  plano do DEM da Fase 5, reaproveitando o próprio modelo de câmera do
  COLMAP (`cam_from_world`/`img_from_cam`). Validado contra cores de
  terreno conhecidas de uma cena sintética com posições de câmera exatas (erro médio de cor < 8/255) —
  processo que expôs e corrigiu um bug real de convenção norte-sul na
  fixture de teste compartilhada, sem afetar as fases anteriores.
  **Resolução sempre independente da resolução do DEM** (bug real
  corrigido em 2026-10-07, encontrado num voo real onde o ortomosaico
  saiu ~100x mais grosseiro que a resolução nativa das fotos): por padrão
  é estimada a partir da altura real de cada câmera acima do terreno e da
  distância focal calibrada, nunca reaproveitando a grade do DEM
  diretamente. **Processado em blocos/tiles** (mesma técnica do Pix4D/
  Metashape — outro bug real corrigido no mesmo dia: em resolução nativa
  sobre uma área real, a grade passou a ter ~168 milhões de pixels e a
  implementação antiga, de uma vez só, estourou a memória). Cada pixel
  vem da foto que o vê mais na vertical (como o modo "Mosaic" do
  Metashape), não da média de todas; a borda vista só de lado é cortada
  (`--max-view-angle`, padrão 30°). Sai como RGB em JPEG, em blocos, com
  pirâmides e máscara interna marcando a área vazia. CLI:
  `htrmapper ortho [--resolution N] [--max-view-angle G]`. GUI: botão
  "Gerar Ortomosaico…". Ver `ARCHITECTURE.md` seções 17 e 26.
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
- **Precisão e verificação** (seção 29): o ajuste limpa os tie points em
  etapas e reotimiza (como a "gradual selection" do Metashape no fluxo
  do USGS); `htrmapper orthometric` grava uma cópia do DEM em altitude
  normal pelo modelo hgeoHNOR2020 do IBGE (o DEM do projeto continua
  elipsoidal, como o ortomosaico precisa); `htrmapper checkpoints` mede
  o erro absoluto do ortomosaico e do DEM contra alvos medidos em campo
  com rover RTK, achando o centro de cada alvo automaticamente.

## Instalação (desenvolvimento)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[gui,cpu,dev]"     # dentro do WSL com GPU: ".[gui,cuda,dev]"
```

O `pycolmap` fica nos extras `cpu` (sem GPU) e `cuda` (`pycolmap-cuda12`),
nunca nos dois: os dois instalam o mesmo módulo, e o de CPU sobrescreveria
o de GPU em silêncio.

## Windows + GPU real (WSL2) — para a Fase 4 (nuvem densa) e além

O `pip install pycolmap` no Windows não tem suporte a CUDA em nenhuma
versão — mesmo com uma GPU NVIDIA real na máquina, a Fase 4 (nuvem densa)
não funciona nativamente no Windows. O caminho recomendado: rodar Fases
1-3 (Importar/Alinhar/Ajustar) na GUI do Windows normalmente, e Fases 4-6
(`htrmapper dense`/`dem`/`ortho`) de dentro do WSL2 com `pycolmap-cuda12`
instalado lá — os caminhos do `projeto.json` são traduzidos
automaticamente entre os dois lados (Windows ↔ `/mnt/c/...`), sem precisar
editar nada à mão. Passo a passo completo e a lógica de detecção em
`ARCHITECTURE.md` seção 23 — **validado em hardware real** (RTX 3060 Ti,
2026-10-06/07); se a virtualização estiver desligada na BIOS (comum em
PCs que nunca usaram máquina virtual), o Windows acusa
`HCS_E_HYPERV_NOT_INSTALLED` ao instalar o WSL2 — a seção 23 documenta
como resolver isso também.

## Uso — processamento automático

Na janela (`htrmapper-gui` ou o atalho do Windows): pasta das fotos, pasta de
saída, nível e **Processar**. Os botões "Escolher pasta…" abrem o diálogo padrão do
Windows; o relatório PDF segue o formato compacto do relatório do Metashape
(6 páginas). Pelo terminal, o mesmo:

```bash
htrmapper processar /caminho/das/fotos --saida /caminho/da/saida --nivel medio
#   --nivel baixo | medio | alto     (padrão: medio, a configuração validada contra o Pix4D)
#   --recomecar                      ignora um processamento anterior na pasta de saída
#   --apagar-trabalho                apaga os dados intermediários ao terminar
```

Na pasta de saída ficam `<nome>_ortomosaico.tif`, `<nome>_dem.tif`,
`<nome>_relatorio.pdf`, `processamento.log` e `projeto.json` (que também
abre no modo manual: menu Avançado → Etapas individuais).

| Nível | Para quê | O que muda |
|---|---|---|
| Baixo | Visualização rápida | menos pontos-chave, nuvem densa leve, ortomosaico com pixel de 2x o GSD |
| Médio | Mapeamento, restituição de linha | a configuração validada contra o Pix4D, pixel = GSD |
| Alto | O mais completo | nuvem densa em resolução 2x maior (DEM/nuvem mais detalhados, bem mais lento) |

## Uso — CLI etapa por etapa

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

# opcionais
htrmapper orthometric projeto.json --output dem_ortometrico.tif   # altitude normal (IBGE hgeoHNOR2020)
htrmapper checkpoints projeto.json --points pontos_rover.csv      # erro contra alvos de campo
```

## Uso — GUI

```bash
htrmapper-gui
```

Abre a tela do processamento automático; as etapas individuais ficam no
menu Avançado. No Windows nativo (sem WSL), a nuvem densa não roda (o
pycolmap do Windows não tem CUDA), então o processamento automático recusa
começar e explica o motivo. No Windows, dá para abrir a GUI com um único duplo-clique em `run_gui.bat`
(ou rodando `run_gui.bat` no `cmd`, dentro da pasta do projeto) — ele cria
o `.venv` se ainda não existir, ativa e roda `pip install -e ".[gui,cpu,dev]"`
sempre (não só na primeira vez, para captar automaticamente uma
dependência nova de uma fase mais recente num `.venv` criado antes dela —
ex. `pycolmap` da Fase 2), antes de chamar `htrmapper-gui`. Depois de um
`git pull`, é só rodar `run_gui.bat` de novo normalmente.

## Otimizador de ortomosaico (QGIS)

`tools/otimizador_ortomosaico_qgis.py` é um script independente para rodar
dentro do QGIS (Terminal Python → Mostrar editor → abrir → Executar). Ele
converte qualquer ortomosaico GeoTIFF RGB 8 bits (do HTRMapper, Pix4D,
Metashape...) num GeoTIFF tiled com pirâmides internas em EPSG:31983, com
compressão escolhida (JPEG q90 por padrão), e valida o resultado. Quando a
entrada já está em EPSG:31983 com resolução automática, os pixels são
copiados 1:1 na mesma grade (sem reamostragem, sem deslocamento). Não usa
nenhum módulo do HTRMapper; precisa só do Python do próprio QGIS.

## Comparar com o Pix4D/Metashape

`tools/comparar_ortomosaicos.py` mede, em centímetros, o deslocamento entre
o nosso ortomosaico e o de outro programa (mesmo voo), por correlação de
fase em janelas de 256 px numa grade de pontos (padrão: a cada 20 m):

```bash
python tools/comparar_ortomosaicos.py ortho_v5.tif ortho_pix4d.tif --saida pontos.geojson
```

Imprime mediana, RMSE, p95 e máximo do deslocamento em E/N, e os pontos
com maior diferença; o GeoJSON abre direto no QGIS (campos `dE_cm`,
`dN_cm`, `d2D_cm`).

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
