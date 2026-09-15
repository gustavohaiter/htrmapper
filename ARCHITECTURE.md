# HTRMapper — Arquitetura e Plano Técnico

Documento de análise obrigatória antes da implementação (ver `PRIMEIRA TAREFA` no
briefing do projeto). Cobre arquitetura, bibliotecas, riscos e a estrutura de
diretórios. A FASE 1 (implementada neste commit) está descrita ao final.

## 1. Premissa central: não reescrever o motor fotogramétrico do zero

Concordo com a orientação do usuário. Um pipeline de fotogrametria completo e
correto (SfM + MVS + bundle adjustment robusto a outliers, com suporte a
rolling shutter, milhões de tie points, etc.) representa o equivalente a
décadas-pessoa de engenharia em produtos maduros (Metashape, Pix4D, RealityCapture,
COLMAP, OpenMVS). Reimplementar tudo do zero é:

- **Tecnicamente arriscado**: bugs sutis em triangulação/BA custam precisão
  geométrica de forma silenciosa — exatamente o que o usuário mais valoriza.
- **Não é onde está o valor diferencial**: o valor deste projeto está na
  integração correta de PPK/RTK como observações ponderadas no ajuste, no
  controle de qualidade quantitativo, e no fluxo de agricultura de precisão
  (SIRGAS2000/UTM, DEM/orthomosaico agrícola). Isso não existe pronto e
  turnkey em nenhuma ferramenta open source.

**Decisão arquitetural**: construir uma aplicação própria (orquestração,
modelo de dados, GNSS weighting, QC, GUI, exportação) sobre um *core*
fotogramétrico composto por bibliotecas maduras, com pontos de extensão
próprios exatamente nos locais que o usuário listou:

```
PPK → pesos GNSS → bundle adjustment → georreferenciamento → DEM → ortomosaico → QC
```

## 2. Bibliotecas candidatas e avaliação

| Componente | Opção escolhida | Alternativas avaliadas | Motivo |
|---|---|---|---|
| Features + matching + SfM (esqueleto) | **COLMAP** (via `pycolmap`, BSD license) | OpenMVG (MPL2, menos ativo), próprio | COLMAP é o SfM/MVS open source mais maduro e validado academicamente; `pycolmap` expõe API Python sobre o C++ nativo (rápido) sem exigir escrevermos CUDA/C++ de baixo nível. Suporta SIFT (GPU/CPU), matching guiado, BA próprio em Ceres. |
| Bundle adjustment com GNSS ponderado | **Ceres Solver, custom cost functions em C++ ou via pycolmap + reconstrução customizada** | g2o (menos manutenção), próprio solver | O COLMAP já usa Ceres internamente, mas o BA *nativo* do COLMAP não aceita pesos GNSS por câmera do jeito que precisamos (ele tem `rig`/prior support limitado). Por isso: usamos o COLMAP para SfM incremental (poses iniciais + tie points), depois **exportamos o problema para um BA próprio em Ceres** (Python bindings `pyceres`, ou binding C++ dedicado) onde adicionamos: (a) resíduo de reprojeção por observação, (b) resíduo de posição GNSS por câmera com covariância do usuário, (c) parâmetros de câmera compartilhados. Este é o componente **mais próprio** do projeto. |
| Álgebra linear | **Eigen** (via pycolmap/pyceres, que já dependem dele) | numpy puro | Eigen é usado nos bindings nativos; não precisamos reimplementar. |
| MVS (nuvem densa) | **OpenMVS** (AGPL-3.0) *ou* **COLMAP dense (patch-match stereo)** (BSD) | | COLMAP dense evita o problema de licença AGPL do OpenMVS (que exigiria que todo o software fosse AGPL se linkado estaticamente). **Decisão: usar COLMAP patch_match_stereo + stereo_fusion como padrão**, deixando OpenMVS como *processo externo opcional* (chamado como binário separado, não linkado) para usuários que preferirem, documentando a implicação de licença. |
| Raster / GeoTIFF / CRS | **GDAL + rasterio** | | Padrão de fato, obrigatório pelo próprio briefing. |
| Transformações geodésicas | **PROJ (via pyproj)** | | Nunca fazer conversão manual lat/lon→UTM. |
| Point cloud (LAS/LAZ, filtragem) | **PDAL** e/ou **laspy** | | PDAL para pipelines de filtragem/classificação; laspy para I/O simples em Python quando PDAL for pesado demais. |
| Visão computacional auxiliar (máscaras, features alternativos) | **OpenCV** | | Padrão, license Apache2/BSD conforme versão. |
| Orquestração, GUI, geoprocessamento leve | **Python 3.11+** | | Produtividade, ecossistema geoespacial. |
| Núcleo de performance própria (GNSS-weighted BA, rolling-shutter futuro) | **C++ com pybind11** quando o protótipo Python/Ceres-python se mostrar insuficiente em desempenho | | Começar em Python (via `pyceres`) para *velocidade de desenvolvimento e corretude*; migrar hot path para C++ só quando perfilado como gargalo real (ver seção de riscos). |
| GUI desktop | **PySide6 (Qt, LGPL)** | Dear ImGui, Electron | Qt é robusto para apps desktop científicos, tem QOpenGLWidget para visualização 3D de nuvens de pontos, e licença LGPL permite uso comercial sem copyleft viral se dinamicamente linkado (regra padrão do PySide6). |
| GPU | **CUDA via COLMAP nativo** quando disponível; fallback CPU sempre | | RTX 3060 Ti detectada e usada automaticamente pelo COLMAP (SIFT GPU, patch-match GPU); nunca obrigatória. |

### Licenças — pontos de atenção
- **OpenMVS é AGPL-3.0**: linkar estaticamente exigiria abrir todo o código sob AGPL.
  Por isso ele é tratado como **processo externo opcional**, nunca dependência
  linkada, e claramente documentado ao usuário antes de habilitar.
- **COLMAP é BSD-3-Clause**: seguro para uso comercial/fechado.
- **Ceres Solver é BSD-3-Clause (Google)**: seguro.
- **GDAL/PROJ/PDAL** são licenças permissivas (MIT/X11 style, BSD).
- **PySide6** é LGPL-3.0 (uso dinâmico é seguro; não modificamos a lib).

Nenhuma linha de código do Metashape é usada, referenciada ou reimplementada
por engenharia reversa. Os métodos aqui descritos (bundle adjustment com
observações GNSS ponderadas, SfM incremental, patch-match MVS) são
literatura fotogramétrica/CV pública (Triggs et al. 2000 "Bundle Adjustment
— A Modern Synthesis"; Schönberger & Frahm, COLMAP 2016; Furukawa & Ponce,
PMVS/CMVS), não propriedade da Agisoft.

## 3. Bundle adjustment com PPK — como será implementado (Fase 3)

Modelo matemático (Gauss-Markov ponderado), consistente com o briefing:

```
minimize  Σ_obs  ρ( ||π(K, R_c, t_c, X_j) - x_ij||² / σ_pixel² )
        + Σ_cam  ||t_c - t_c^GNSS||_W_c²
```

Onde:
- `π` é a projeção pinhole + distorção (Brown-Conrady: k1,k2,k3,p1,p2, depois k4/b1/b2);
- `ρ` é uma *loss function* robusta (Huber/Cauchy) para tie points, essencial
  porque matches espúrios sobrevivem ao RANSAC inicial;
- `t_c^GNSS` é a posição da câmera vinda do PPK, no CRS do projeto (já
  transformada geodesicamente, nunca por conversão linear ad-hoc);
- `W_c = diag(1/σ_x², 1/σ_y², 1/σ_z²)` é a matriz de peso **derivada
  diretamente da Camera Accuracy configurada pelo usuário** (σ_xy para X,Y;
  σ_z para Z, no referencial local ENU da câmera, depois rotacionado para o
  CRS do projeto).

Isso é literalmente uma "soft constraint"/observação com peso, não uma
âncora fixa: câmeras podem se mover livremente durante o ajuste até o ponto
em que o gradiente do erro de reprojeção supera o gradiente do erro GNSS
ponderado — exatamente o comportamento de um ajuste geodésico clássico
(idêntico em espírito ao que Metashape/Pix4D fazem internamente, mas
implementado com fórmulas nossas, não copiadas).

Implementação técnica: cost function customizada em Ceres
(`ceres::AutoDiffCostFunction` ou `ceres::CostFunction` analítico) com dois
tipos de resíduo (`ReprojectionError`, `GnssPositionError`), registrada via
`pyceres` (bindings Python para Ceres) para manter velocidade de
desenvolvimento; migração para C++ puro fica reservada para quando o
profiling mostrar que a camada Python é o gargalo (ver riscos).

σ (sigma) nunca é zero: mesmo com "0.00" no futuro, sempre há um piso
numérico mínimo para evitar matriz de peso singular — isso será documentado
e testado explicitamente (ver testes de Fase 3).

### Achado da Fase 2 relevante para a Fase 3: `PosePrior` do próprio COLMAP

Durante a implementação da Fase 2, validamos interativamente (via `pycolmap`
4.2) que o COLMAP já possui uma tabela `PosePrior` no seu banco de dados,
com `position` + `position_covariance` + `coordinate_system` (WGS84 ou
Cartesian) por imagem, populada automaticamente a partir do GPS do EXIF. Ela
já é o mecanismo que `match_spatial` usa para restringir pares candidatos
(Fase 2), e o COLMAP expõe também `IncrementalPipelineOptions.use_prior_position`
e uma classe `PosePriorBundleAdjustmentOptions`/`CeresPosePriorBundleAdjustmentOptions`
— ou seja, um bundle adjustment com prior de posição **ponderado por
covariância** já existe dentro do Ceres embutido no COLMAP.

Validamos experimentalmente que dá para sobrescrever a covariância desse
`PosePrior` com a nossa própria (`CameraAccuracy` convertida para
`diag(σ_xy², σ_xy², σ_z²)`) via `Database.update_pose_prior()`, no mesmo
registro que o COLMAP já criou a partir do EXIF. Isso confirmou que dá
para usar o bundle adjustment de prior de posição **já existente no
COLMAP** (`pycolmap.create_pose_prior_bundle_adjuster`), em vez de
escrever uma `ceres::CostFunction` própria do zero via `pyceres` — exatamente
a abordagem usada na implementação da Fase 3 (ver seção 14).

**Resolução da dúvida em aberto (referencial da `position_covariance`):**
como o `coordinate_system=WGS84` do COLMAP não documenta explicitamente se
a covariância é interpretada em graus² ou metros² quando convertida
internamente, a decisão de projeto foi **não depender dessa conversão
implícita**. A Fase 3 constrói os `PosePrior` sempre com
`coordinate_system=CARTESIAN`, com a posição já transformada para o CRS
do projeto (metros) pelo nosso próprio `geo.crs` — o mesmo transform que a
Fase 2 usa para georreferenciar. Assim a covariância é inequivocamente em
metros², no mesmo referencial em que a reconstrução já vive (alinhada
pela Fase 2 via `align_reconstruction_to_locations`), sem depender de
nenhuma conversão de unidade não documentada do COLMAP. Isso foi validado
empiricamente com viés conhecido (ver seção 14).

## 4. Reconstrução densa (Fase 4)

Pipeline: `undistort images (usando parâmetros calibrados do BA) → patch
match stereo (COLMAP) → stereo fusion → nuvem de pontos densa
georreferenciada (transformação usando o mesmo CRS do projeto)`.
Níveis de qualidade (baixa/média/alta/muito alta) mapeiam para os
parâmetros nativos do COLMAP: `max_image_size`, `window_radius`,
`num_samples`, `geom_consistency`. Fallback CPU existe (mais lento) porque o
patch-match do COLMAP roda em CUDA *ou* CPU.

## 5. DEM/DSM (Fase 5)

A partir da nuvem densa (que já está no CRS do projeto, em metros):
1. Filtragem de outliers estatística (PDAL `outlier` filter, SOR/statistical).
2. Rasterização por interpolação (IDW ou TIN→raster) numa grade cuja
   resolução default é derivada do GSD médio calculado a partir de
   altitude de voo, focal length e tamanho de pixel do sensor (nunca um
   valor mágico fixo) — usuário pode sobrepor manualmente.
3. Escrita via GDAL garantindo: CRS (`SetProjection`), `GeoTransform`
   correto (origem = canto superior-esquerdo, pixel size correto,
   sem rotação salvo casos especiais), `NoData` explícito.

## 6. Ortomosaico (Fase 6)

Ortorretificação verdadeira (true-ortho quando computacionalmente viável):
para cada pixel de saída, usar o DSM para achar a altura do terreno,
projetar de volta para a imagem de origem via os parâmetros de câmera
calibrados (intrínsecos + extrínsecos do BA), escolher a imagem "melhor"
por critério de ângulo de visada/distância ao centro, e aplicar blending
com feathering nas seamlines. Isso evita o erro clássico de ortomosaicos
"baratos" que usam só um plano médio (double-mapping em árvores/estruturas).//
Implementação inicial usará GDAL (`gdal.Warp` com função de altura por
pixel custom via VRT + numpy, ou rasterização direta por ray-casting nas
poses calibradas) — este é outro ponto de desenvolvimento próprio genuíno.

## 7. Validação contra o Metashape (Fase 8)

Metodologia:
1. Processar o **mesmo conjunto de imagens brutas** nos dois softwares, com
   a mesma Camera Accuracy informada.
2. Exportar de ambos: posições de câmera (CSV com X,Y,Z,yaw,pitch,roll +
   erro por câmera), nuvem de pontos densa (LAS), DEM (GeoTIFF), ortomosaico
   (GeoTIFF).
3. Script de comparação (`tools/compare_metashape.py`, a implementar na
   Fase 8) que calcula: RMSE X/Y/Z das posições de câmera pareadas por
   nome de arquivo; diferença de DEM célula-a-célula (após reamostragem
   para a mesma grade) com estatísticas (média, desvio-padrão, percentis,
   mapa de diferença); diferença espacial de ortomosaico via correlação de
   NDVI/RGB e deslocamento sub-pixel (phase correlation); comparação de
   área calculada do polígono de cobertura.
4. Relatório HTML/PDF com todas as métricas lado a lado. **Nunca declarar
   "precisão equivalente" sem esse relatório gerado e anexado.**

## 8. Riscos técnicos

1. **Overhead de bindings Python (`pyceres`, `pycolmap`) em datasets grandes
   (milhares de imagens)** — mitigação: perfilar cedo; ter plano B de mover
   a montagem do problema de BA para C++ nativo com pybind11 caso o overhead
   de marshalling Python→C++ domine o tempo total.
2. **Rolling shutter da DJI Mavic 3M**: o sensor é CMOS rolling shutter;
   em voos com alta velocidade angular (viradas rápidas) isso introduz
   distorção geométrica não modelada por um pinhole estático. Mitigação
   inicial (Fase 1-6): orientar usuário a voar em trajetórias suaves e
   validar reprojection error por câmera; modelagem completa de rolling
   shutter (spline de pose por tempo de leitura de linha) fica para uma
   fase pós-MVP, arquitetura já preparada (timestamp de exposição por
   imagem e id de linha de leitura ficam no modelo de dados desde a Fase 1).
3. **Falta de GCP (uso típico do usuário)**: sem GCPs, a precisão absoluta
   do produto final depende inteiramente da qualidade do PPK e da
   consistência do offset da antena GNSS→centro óptico (lever arm) e do
   boresight (desalinhamento entre IMU/gimbal e câmera). Isso deve ser
   modelado explicitamente (parâmetros de lever arm/boresight,
   inicialmente fixos/calibráveis manualmente, com plano de auto-calibração
   futura) — do contrário, a precisão relativa (forma do terreno) pode ser
   boa mas a precisão absoluta (posição) terá um viés sistemático.
4. **Paridade de precisão com Metashape**: Metashape usa um BA proprietário
   refinado por ~15 anos; atingir paridade exata de RMSE não é garantido
   nem prometido — o objetivo declarado é *medir* a diferença
   quantitativamente (Fase 8), não assumir equivalência.
5. **Licença AGPL do OpenMVS**: risco de contaminação de licença se alguém
   linkar estaticamente no futuro; mitigado por mantê-lo como processo
   externo opcional, nunca dependência do pacote principal.
6. **Qualidade de matching em áreas agrícolas homogêneas** (plantações
   uniformes, baixa textura) — cenário adversarial conhecido para SfM.
   Mitigação: usar prior GNSS/temporal para restringir pares candidatos
   (reduz falsos positivos em texturas repetitivas) e permitir overlap
   configurável mais alto para essas áreas.
7. **Textura periódica (fileiras de plantio)** — risco distinto do item 6,
   confirmado visualmente numa foto real de talhão compartilhada durante o
   desenvolvimento: fileiras paralelas e repetitivas (cana, milho, soja em
   linha) são visualmente ricas — SIFT encontra muitos keypoints — mas
   *ambíguas*: um ponto na fileira N pode casar erroneamente com o ponto
   equivalente na fileira N+1, gerando outliers *sistemáticos* (deslocamento
   pequeno e consistente na direção perpendicular às fileiras), que o
   RANSAC da verificação geométrica nem sempre rejeita bem quando o erro é
   pequeno. Mitigação primária: o matching restrito por posição GNSS
   (`match_spatial`, Fase 2) já reduz o espaço de busca à vizinhança
   espacial plausível, o que descarta a maioria dos falsos candidatos
   "fileira errada" antes mesmo da verificação geométrica. Se essa mitigação
   se mostrar insuficiente em dados reais, o próximo recurso é reduzir
   `spatial_max_distance_m` (menos candidatos, mais próximos) ou aumentar o
   overlap de voo recomendado ao usuário para essas culturas.

## 9. Partes mais difíceis de igualar ao Metashape

- **Ortomosaico true-ortho com seamlines otimizadas** (grafo de corte
  minimizando artefatos visíveis) — é uma área com muito tuning heurístico
  acumulado em produtos comerciais.
- **Robustez do BA em datasets muito grandes** (>2000 imagens) mantendo
  tempo de processamento aceitável — exige BA hierárquico/incremental bem
  ajustado (COLMAP tem isso, mas nossa camada de pesos GNSS customizada
  precisa preservar essa escalabilidade).
- **Auto-calibração de lever-arm/boresight** sem GCP — Metashape/Pix4D têm
  anos de heurísticas proprietárias para isso; nosso caminho será
  calibração explícita assistida (voo calibrável + otimização opcional no
  BA) em vez de mágica automática.

## 10. Estrutura de diretórios (implementada)

```
htrmapper/
├── ARCHITECTURE.md
├── README.md
├── pyproject.toml
├── src/htrmapper/
│   ├── core/         # modelo de projeto, formato de arquivo, logging
│   ├── io/           # EXIF/XMP, importação de imagens
│   ├── geo/          # CRS, transformações geodésicas (PROJ/GDAL)
│   ├── gnss/         # precisão GNSS, pesos para o BA (fundação p/ Fase 3)
│   ├── gui/          # interface desktop (PySide6) — mínima na Fase 1
│   ├── cli/          # CLI para uso sem GUI / automação / CI
│   └── (futuro) sfm/, mvs/, dem/, ortho/, qc/, export/, rollingshutter/
├── tests/            # pytest, dados sintéticos (sem depender de imagens reais)
└── data/samples/     # espaço para datasets reais de teste (git-ignored)
```

## 11. Especificação do relatório de processamento (Quality Report)

O usuário forneceu um relatório real do Agisoft Metashape (voo eBee,
câmera S.O.D.A, 404 imagens) como referência do que o HTRMapper deve
produzir ao final do processamento. A tabela abaixo mapeia **cada campo**
desse relatório para o modelo de dados do HTRMapper e para a fase que o
calcula, para deixar explícito que o objetivo não é copiar o layout do
Metashape, mas chegar às mesmas grandezas fotogramétricas por meios
próprios, e nunca reportar um número sem tê-lo de fato calculado.

| Seção do relatório (Metashape) | Campo | Fonte no HTRMapper | Fase |
|---|---|---|---|
| Capa | Preview do ortomosaico | Miniatura do GeoTIFF final | 6 |
| Survey Data | Número de imagens / estações de câmera | `len(project.images)` | 1 (já disponível) |
| Survey Data | Mapa de posições de câmera + overlap | `geo.crs` + poses; overlap requer grafo de matches | 1 (posições) / 2 (overlap) |
| Survey Data | Altitude de voo | `drone-dji:RelativeAltitude` (XMP) ou EXIF GPS altitude | 1 (já disponível) |
| Survey Data | Resolução em solo (GSD, cm/pix) | `geo.gsd.estimate_gsd_cm` (focal, altitude, sensor via crop factor) | 1 (já disponível, estimativa pré-BA) |
| Survey Data | Área de cobertura (km²) | `geo.coverage` (hull convexo das posições de câmera) | 1 (estimativa) / 6 (área real do ortomosaico) |
| Survey Data | Tie points / Projections | Saída do SfM (COLMAP) | 2 |
| Survey Data | Reprojection error (pix) | Saída do bundle adjustment | 3 |
| Survey Data | Tabela de câmeras (modelo, resolução, focal, pixel size, precalibrada) | `ImageRecord` agregado por `camera_model` | 1 (já disponível, exceto "precalibrada") |
| Camera Calibration | Resíduos de imagem por modelo de câmera | Saída do BA (resíduos de reprojeção por observação) | 3 |
| Camera Calibration | Coeficientes (f, cx, cy, k1-k4, p1, p2, b1, b2) + matriz de correlação | Parâmetros otimizados do BA + matriz de covariância do solver | 3 |
| Camera Locations | Mapa de elipses de erro (X,Y,Z) | Resíduo `posição_ajustada - posição_GNSS` por câmera, decomposto e escalado pela covariância a posteriori | 3 |
| Camera Locations | Tabela de erro médio (X, Y, Z, XY, Total em cm) | RMSE dos resíduos GNSS acima | 3 |
| DEM | Preview colorido + resolução + densidade de pontos | Raster gerado + estatística da nuvem densa | 4 e 5 |
| Orthomosaic | Preview + tamanho + CRS | GeoTIFF final | 6 |
| Processing Parameters → General | Cameras / Aligned cameras / CRS / Rotation angles | `Project` + resultado do SfM | 1 (CRS) / 2 (alinhamento) |
| Processing Parameters → Tie Points | Points, RMS/Max reprojection error, key point size, parâmetros de alinhamento (accuracy, key point limit, tie point limit) | Config de features/matching + saída do SfM/BA | 2 e 3 |
| Processing Parameters → Depth Maps | Quality, filtering mode, tempo/memória | Config e telemetria do MVS (patch-match) | 4 |
| Processing Parameters → Point Cloud | Contagem de pontos, atributos, tempo/memória | Saída do MVS/fusion | 4 |
| Processing Parameters → DEM | Tamanho, CRS, parâmetros de reconstrução, tempo/memória | Config e telemetria da rasterização | 5 |
| Processing Parameters → Orthomosaic | Tamanho, CRS, blending mode, surface, tempo/memória | Config e telemetria da ortorretificação | 6 |
| Processing Parameters → System | SO, RAM, CPU, GPU(s) | `core.system_info` (detecção real de hardware) | 1 (já disponível) |

### Regra de honestidade do relatório

Qualquer campo cuja fase produtora ainda não foi implementada é exibido
explicitamente como **"Não disponível — calculado na Fase N"**, nunca como
zero, `null` silencioso ou um valor inventado. Isso é obrigatório para
respeitar a regra 3 do briefing ("não diga que algo possui precisão
centimétrica sem validação"). O relatório do HTRMapper cresce
incrementalmente: a cada fase implementada, mais seções passam de
"pendente" para valores reais.

### Implementado nesta rodada (fundação do relatório, Fase 1)

- `geo.gsd`: estimativa de GSD (cm/pixel) a partir de altitude, focal
  length e a largura do sensor **derivada do fator de crop**
  (`sensor_width_mm = 36mm / crop_factor`, `crop_factor =
  focal_length_35mm_equiv / focal_length_mm`) — evita depender de um banco
  de dados de sensores por modelo de câmera, que nem sempre está
  disponível/atualizado.
- `geo.coverage`: área de cobertura por hull convexo das posições de
  câmera no CRS do projeto (aproximação pré-alinhamento; a área real do
  ortomosaico substitui isso na Fase 6).
- `core.system_info`: detecção real de CPU, RAM, GPU NVIDIA/CUDA e VRAM
  (via `nvidia-smi`, com fallback correto quando não há GPU).
- `core.report`: modelo de dados do relatório completo (todas as seções
  da tabela acima) e um renderizador HTML que popula o que já é
  calculável na Fase 1 e marca o resto como pendente, com a fase exata
  que o produzirá.

## 12. Fase 1 — o que está implementado neste commit

- `core.project`: modelo de projeto (`Project`, `ImageRecord`,
  `GnssAccuracyConfig`) serializável em JSON, com versionamento de schema,
  salvar/carregar.
- `io.exif_reader`: leitura robusta de EXIF (GPS lat/lon/alt, orientação,
  focal length, modelo/fabricante da câmera, timestamp, dimensões),
  tolerante a campos ausentes.
- `io.xmp_reader`: extração do pacote XMP embutido no JPEG e parsing dos
  campos DJI relevantes (`drone-dji:GimbalYawDegree/PitchDegree/RollDegree`,
  `FlightYawDegree/PitchDegree/RollDegree`, `RelativeAltitude`,
  `AbsoluteAltitude`, `RtkFlag`, `RtkStdLon/Lat/Hgt`, `GpsStatus`).
- `io.image_import`: varredura de pasta, construção de `ImageRecord` por
  imagem, geração do relatório de importação (contagens, imagens sem
  coordenadas, coordenadas inválidas, extensão espacial, altitude
  min/max, câmeras detectadas).
- `geo.crs`: wrapper sobre `pyproj` para transformação geodésica
  correta WGS84 → CRS do projeto (ex. EPSG:31983), nunca conversão manual.
- `gnss.accuracy`: `CameraAccuracy` (σ_xy, σ_z) e cálculo de matriz de peso
  (`1/σ²`), fundação matemática que será usada pelo bundle adjustment
  na Fase 3.
- `gui.main_window`: visualizador mínimo (PySide6) com tabela de imagens
  importadas e mapa de posições de câmera no CRS do projeto — real, a
  partir dos dados importados (não é mockup: usa o resultado de
  `image_import` de fato).
- `cli.main`: comando `htrmapper import <pasta> --epsg 31983` para uso
  sem GUI / CI / automação.
- Testes automatizados com **dados sintéticos gerados em runtime**
  (JPEGs com EXIF+XMP DJI fabricados nos testes) cobrindo: parsing EXIF,
  parsing XMP DJI, importação de pasta (incluindo imagens sem GPS),
  transformação de CRS (valores conhecidos), pesos GNSS, e
  save/load de projeto.
- `geo.gsd`, `geo.coverage`, `core.system_info`, `core.report`: fundação
  do relatório de qualidade (ver seção 11 acima), incluindo exportação
  HTML via `htrmapper import --report-out relatorio.html`.

## 13. Fase 2 — features, matching e SfM inicial (implementada)

Construída inteiramente sobre `pycolmap` (bindings Python do COLMAP),
conforme a decisão arquitetural da seção 1: nenhum detector de features,
matcher ou solver de SfM é reimplementado aqui.

- `sfm.camera_model`: deriva a estimativa inicial de intrínsecos
  (fx, fy, cx, cy) via o mesmo truque de crop-factor do `geo.gsd`
  (`sensor_width_mm_from_crop_factor`, reaproveitado, não duplicado),
  agrupando imagens por `camera_model` — cada câmera física ganha sua
  própria estimativa, preparando o suporte a múltiplas câmeras (RGB +
  multiespectral no futuro). Levanta `InsufficientExifError` de forma
  explícita (nunca adivinha) quando faltam os campos EXIF necessários;
  nesse caso o `sfm.pipeline` cai para o modo automático do próprio
  COLMAP (`CameraMode.AUTO` + leitura de EXIF nativa), documentado como
  tal no resultado.
- `sfm.pipeline.run_structure_from_motion`: extração de features SIFT
  (`pycolmap.extract_features`, limite de key points configurável,
  default 40.000 por imagem), matching espacial restrito pelas posições
  GNSS (`pycolmap.match_spatial`, que já consome o `PosePrior` que o
  COLMAP cria automaticamente a partir do GPS do EXIF — ver achado na
  seção 3), com verificação geométrica/RANSAC embutida
  (`TwoViewGeometryOptions`); cai para matching exaustivo quando há menos
  de 3 imagens com posição GNSS válida (GNSS nunca é dependência
  absoluta, conforme o briefing). Reconstrução incremental via
  `pycolmap.incremental_mapping`. Georreferenciamento por alinhamento de
  similaridade (`pycolmap.align_reconstruction_to_locations`) às mesmas
  posições GNSS, transformadas para o CRS do projeto via `geo.crs` —
  nunca um reescalonamento linear ingênuo. Falhas parciais (imagens não
  registradas, componentes desconectados, alinhamento que falha) são
  reportadas explicitamente no `SfmResult`, nunca escondidas.
- `core.project.SfmSummary`: persiste o resumo do resultado da Fase 2 no
  arquivo de projeto (reprodutibilidade), mantendo a reconstrução
  completa (tie points, observações) no formato nativo do COLMAP em
  disco, carregável diretamente por `pycolmap` nas fases seguintes.
- `core.report`: quando `project.sfm` existe, preenche "Tie points",
  "Projections", "Câmeras alinhadas" e um novo campo
  **"Reprojection error (inicial, SfM/COLMAP)"**, mantido explicitamente
  distinto do "Reprojection error (final, ponderado por GNSS)" — que
  continua pendente da Fase 3 — para nunca confundir o erro do BA interno
  não ponderado do COLMAP com o resultado final ponderado por GNSS que o
  usuário pediu.
- CLI: `htrmapper align <projeto.json> --workdir <pasta>`. GUI: botão
  "Alinhar (SfM)…", que atualiza o mapa de posições de câmera para
  mostrar as posições **alinhadas pelo SfM** (rótulo distinto do mapa de
  GNSS bruto da Fase 1).

### Validação real (não apenas "não quebrou")

Como imagens sintéticas de cor sólida (usadas nos testes da Fase 1) têm
zero textura — SIFT não encontra nenhum keypoint nelas — foi criado um
gerador de cena sintética texturizada (`tests/synthetic_scene.py`): uma
"textura de terreno" com milhares de formas coloridas aleatórias,
fotografada em nadir (câmera olhando reto para baixo, sem rotação) a
partir de posições de câmera conhecidas, simulando exatamente a geometria
pinhole via recorte+reamostragem da textura na escala implicada pela
fórmula de GSD — sem necessidade de um motor de renderização 3D, porque a
cena é um plano e a câmera não tem inclinação.

Isso permitiu um teste de integração real, com verdade de campo conhecida
(`tests/test_sfm_pipeline.py`): todas as imagens se registram, ~1000 tie
points triangulados, erro de reprojeção médio de ~0,085 pixel (consistente
com uma cena sintética sem ruído), e — mais importante — as posições de
câmera recuperadas após o alinhamento por similaridade às coordenadas
GNSS batem com as posições verdadeiras conhecidas a menos de 1 metro
(na prática, submilimétrico). Isso valida de ponta a ponta: extração de
features, matching restrito por GNSS, SfM incremental, e georreferenciamento
— não apenas que o pipeline "não quebrou".

## 14. Fase 3 — bundle adjustment ponderado por GNSS/PPK (implementada)

Este é o componente mais central do projeto (ver seção 1). Implementado
sobre o bundle adjuster de prior de posição do próprio COLMAP
(`pycolmap.create_pose_prior_bundle_adjuster`), descoberto e validado na
Fase 2 (seção 3) — não uma cost function própria em Ceres via `pyceres`,
que era o plano original antes desse achado.

- `ba.weighted_bundle_adjustment.run_gnss_weighted_bundle_adjustment`:
  carrega a reconstrução já georreferenciada da Fase 2, constrói um
  `PosePrior` por câmera com `coordinate_system=CARTESIAN` (posição no CRS
  do projeto, via `geo.crs`, nunca WGS84 direto — ver seção 3) e
  `position_covariance = diag(σ_xy², σ_xy², σ_z²)` a partir da
  `CameraAccuracy` do projeto. Roda o solver e devolve o resíduo GNSS
  por câmera (posição ajustada − posição GNSS) e o RMSE X/Y/Z/XY/Total em
  centímetros — exatamente os campos que faltavam na tabela "Camera
  Locations" do relatório.

### Validação decisiva: o peso é matemático, não decorativo

Testado injetando um viés conhecido (5 m) na posição GNSS de uma câmera e
rodando o ajuste com duas configurações de `CameraAccuracy`:

- **Sigma apertado (0,02 m)**: a posição resolvida ficou a poucos
  centímetros da observação GNSS (errada) — o peso dominou.
- **Sigma frouxo (5,0 m)**: a maior parte do viés de 5 m permaneceu como
  resíduo não absorvido — a reprojeção dominou, o GNSS ruim foi
  corretamente desconfiado.

Isso é exatamente o comportamento de uma observação ponderada (Gauss-
Markov), nunca uma âncora rígida — a exigência central do briefing (regras
4, 5 e 6). Ver `tests/test_weighted_bundle_adjustment.py`.

### Achado durante a validação: degenerescência de autocalibração em voo nadir plano

Ao permitir refinar `f, cx, cy` simultaneamente com o prior de posição
GNSS numa cena sintética plana e nadir (altitude constante, sem imagens
oblíquas), a focal length divergiu de ~232px para 580-710px, mesmo com o
erro de reprojeção continuando baixo (~0,06px) — uma degenerescência
clássica de autocalibração fotogramétrica: para uma cena plana, `f` e a
elevação (não observada) dos tie points são conjuntamente não-identificáveis,
e um prior de posição de câmera não resolve isso porque não restringe a
elevação da cena. Fixar os intrínsecos (mantendo os já refinados pela
Fase 2) convergiu de forma limpa e estável. Por isso `BaConfig.refine_intrinsics`
é `False` por padrão — só deve ser ligado quando o voo tem diversidade
geométrica real (imagens oblíquas, altitude variável, GCPs), conforme a
regra do briefing de não otimizar parâmetros sem evidência de que são
observáveis.

- `core.report`: quando `project.ba` existe, preenche "Camera Locations"
  (RMSE X/Y/Z/XY/Total em cm) e "Camera Calibration" (parâmetros do
  modelo de câmera pós-ajuste), e o "Reprojection error (final, ponderado
  por GNSS)" da Survey Data — mantido distinto do valor inicial da Fase 2.
  A matriz de correlação dos coeficientes de calibração (como no relatório
  Metashape de referência) ainda não está implementada — fica marcada
  como tal, nunca inventada.
- CLI: `htrmapper adjust <projeto.json> --workdir <pasta>`. GUI: botão
  "Ajustar (GNSS)…", que atualiza o mapa de câmeras para mostrar as
  posições pós-ajuste.

## 15. Fase 4 — nuvem de pontos densa (implementada, com limitação de ambiente)

Construída sobre o próprio pipeline denso do COLMAP —
`pycolmap.undistort_images` → `pycolmap.patch_match_stereo` →
`pycolmap.stereo_fusion` — sem nenhum algoritmo de MVS reimplementado,
conforme a decisão da seção 4.

- `mvs.dense.run_dense_reconstruction`: desdistorce as imagens usando os
  intrínsecos já calibrados (Fase 2/3), roda patch-match stereo e fusão
  estéreo, e exporta a nuvem resultante para **LAS** (via `laspy`), com o
  CRS do projeto embutido no cabeçalho (`LasHeader.add_crs`), cores RGB
  reais dos pontos, e escala de 0,1mm — nunca mais grosseiro do que a
  precisão que o resto do pipeline afirma ter.
- Níveis de qualidade (baixa/média/alta/muito_alta), conforme pedido no
  briefing: mapeados para `max_image_size` do COLMAP (o parâmetro nativo
  de tradeoff compute/qualidade do patch-match), com "muito_alta"
  processando na resolução nativa e cada nível abaixo reduzindo a
  resolução linear pela metade — a mesma curva de tradeoff usada pelas
  ferramentas de referência. `geom_consistency` (verificação de
  consistência entre vistas) é desabilitado apenas no nível mais baixo.

### Limitação de ambiente confirmada empiricamente (não hipotética)

`pycolmap.patch_match_stereo` **não tem fallback de CPU nativo** — requer
GPU CUDA (NVIDIA) ou HIP (AMD); sem uma delas, o próprio COLMAP levanta
`ValueError: Dense stereo reconstruction requires CUDA or HIP...`. Isso
foi confirmado interativamente durante o desenvolvimento (nenhum erro
foi assumido sem reproduzir). Este módulo checa `pycolmap.has_cuda` antes
de iniciar e falha rápido com uma mensagem clara, em vez de deixar esse
erro nativo aparecer no meio de uma etapa longa.

**O ambiente de desenvolvimento usado neste projeto não possui GPU.**
Isso significa que a etapa de cálculo denso em si (patch-match stereo +
fusão) **não pôde ser executada nem validada numericamente aqui** — só o
caminho de erro (falta de GPU) e a exportação LAS (testados com uma
nuvem de pontos construída manualmente, sem depender do cálculo denso em
si) puderam ser validados de fato. O usuário tem uma RTX 3060 Ti (ver
seção 17 do briefing original) — a validação completa desta fase, com
números reais de reconstrução densa, só pode acontecer na máquina do
usuário. Isso é dito aqui explicitamente para não alegar uma validação
que não ocorreu (regra 3 do briefing).

O fallback de CPU para máquinas sem GPU permanece **não implementado**:
a arquitetura já reserva o OpenMVS como processo externo opcional (seção
2, por causa da licença AGPL), mas integrá-lo (conversão do formato de
cena, chamada do binário `DensifyPointCloud`) fica para uma iteração
futura, priorizado apenas se o usuário precisar processar em uma máquina
sem GPU.

- `core.report`: quando `project.mvs` existe, preenche "Point Cloud" nos
  Processing Parameters e uma estimativa de densidade de pontos em
  "Digital Elevation Model" (usando a área do hull convexo de câmeras da
  Fase 1 como proxy, já que a área real do ortomosaico só existe na
  Fase 6 — rotulado como tal).
- CLI: `htrmapper dense <projeto.json> --workdir <pasta> --quality media`.
  GUI: botão "Nuvem densa…".

## 16. Fase 5 — geração de DEM/DSM (implementada e validada)

Rasteriza a nuvem de pontos densa da Fase 4 num raster de elevação
georreferenciado. É um **DSM** (superfície, não terreno nu): a nuvem
vem de MVS baseado em imagem, então cada ponto já é uma amostra de
"primeira superfície" (dossel, estruturas, solo) — não existe
classificação solo/não-solo como em LiDAR para separar aqui. A
classificação de solo para um DEM de terreno nu propriamente dito
continua futura, conforme o próprio briefing ("classificação de solo
futuramente").

Construída sobre GDAL (via `rasterio`) para a escrita do GeoTIFF e
`scipy.interpolate` para a interpolação — nenhum algoritmo de
triangulação/rasterização próprio.

- `dem.generation.run_dem_generation`:
  1. **Filtro de outliers robusto (MAD)**: mediana + desvio absoluto
     mediano, não um corte de percentil fixo. Um corte de percentil fixo
     sempre remove a mesma fração (~0,1%) dos extremos, *independente* de
     quantos outliers reais existem — um cluster de vários pontos
     correlacionados e errados na maior parte sobrevive a ele, porque
     percentil mede posição no ranking, não a distância real à massa de
     dados. Isso foi um bug real pego pelo próprio teste (ver abaixo),
     corrigido antes de chegar a produção.
  2. **Resolução automática**: `espaçamento_médio × 2,5`, onde
     `espaçamento_médio = sqrt(área / número_de_pontos)`. Heurística
     documentada como tal (não uma lei física): uma célula do DEM precisa
     de suporte de múltiplos pontos da nuvem densa para interpolar de
     forma confiável, e a prática comum em fotogrametria aérea recomenda
     resolução de DEM/ortomosaico de 2-4x o espaçamento de pontos/GSD.
     Sempre sobrescrevível pelo usuário (`DemConfig.resolution_m`).
  3. **Interpolação + preenchimento de buracos**: `scipy.interpolate.griddata`
     linear dentro do hull convexo dos pontos, depois nearest-neighbor
     para preencher o que sobrar (fora do hull, ou buracos que a
     interpolação linear não alcança) — nunca extrapolação linear sem
     controle.
  4. Escrita do GeoTIFF via `rasterio`: CRS do projeto (nunca hardcoded),
     geotransform norte-para-cima correto, NoData explícito (-9999),
     Float32, escala compatível com a precisão centimétrica do resto do
     pipeline.

### Validação com verdade de campo conhecida

Como a Fase 4 (nuvem densa real) não pôde ser executada nesta máquina
(sem GPU — ver seção 15), a validação da Fase 5 usa uma nuvem de pontos
sintética construída diretamente a partir de uma função de terreno
conhecida (plano suave + ondulação de baixa amplitude e grande
comprimento de onda), amostrada com ruído realista (1cm). O DEM gerado
é comparado ponto a ponto contra o valor **verdadeiro** da função em
centenas de posições aleatórias: erro médio absoluto abaixo de 15cm,
erro máximo abaixo de 50cm — não apenas "rodou sem erro", mas "os
números batem com a superfície real conhecida". Também validado:
resolução automática vs. definida pelo usuário, geotransform/NoData
corretos, ausência de buracos na saída, e — o achado do filtro de
outliers acima — remoção exata de um número conhecido de pontos
injetados como erro grosseiro.

- `core.report`: quando `project.dem` existe, preenche "Digital
  Elevation Model" (resolução real, densidade de pontos real da nuvem
  densa) e o campo DEM dos Processing Parameters.
- CLI: `htrmapper dem <projeto.json> --output <dem.tif> [--resolution N] [--no-filter]`.
  GUI: botão "Gerar DEM…".

## 17. Fase 6 — geração de ortomosaico (implementada e validada)

**Isto é ortorretificação 2.5D, não true-ortho 3D completo** — declarado
explicitamente no docstring do módulo, conforme a regra do projeto contra
superestimar precisão. Para cada pixel de saída, a altura do terreno vem
do DEM por-pixel (Fase 5), então relevo de terreno é tratado corretamente
(diferente do "plano médio único" que o briefing original explicitamente
alerta contra). O que **não** é feito: teste de visibilidade/oclusão
por-pixel (z-buffer) contra toda a cena — um objeto alto entre uma câmera
e um ponto mais baixo próximo ainda pode se projetar incorretamente sobre
o pixel desse ponto mais baixo ("vazamento"/duplo-mapeamento ao redor de
feições verticais). É exatamente essa checagem de visibilidade completa
que separa um true-ortho 3D de verdade (ray-casting com teste de
visibilidade por pixel) do que está implementado aqui.

Construído reaproveitando diretamente o modelo de câmera do próprio
COLMAP (`Image.cam_from_world()`, `Camera.img_from_cam()` — a mesma
projeção usada no erro de reprojeção em todo o projeto), amostrando das
imagens **não-distorcidas** que a etapa `pycolmap.undistort_images` da
Fase 4 já produziu (assim a amostragem nunca precisa lidar com distorção
de lente separadamente). Nenhuma matemática de câmera própria, nenhuma
I/O de raster própria (`rasterio`/GDAL) — conforme a decisão de
arquitetura de construir sobre bibliotecas maduras.

- `ortho.orthomosaic.run_orthomosaic_generation`:
  1. Carrega a grade do DEM (coordenadas de mundo por pixel via os
     coeficientes do transform afim diretamente — não
     `rasterio.transform.xy`, que não preserva a forma de arrays 2D).
  2. Para cada câmera registrada da reconstrução: projeta todos os pontos
     3D válidos do terreno (`cam_from_world()` seguido de `img_from_cam(...,
     check_cheirality=True)`), filtra por NaN (pontos atrás da câmera) e
     por limites da imagem, e amostra a cor por interpolação bilinear.
  3. **Blending por distância-à-borda ("feathering")**: cada câmera
     contribui um voto ponderado por pixel de saída, com peso decrescendo
     linearmente perto da borda da imagem-fonte — assim a costura entre
     duas imagens sobrepostas suaviza em vez de mostrar uma borda dura.
     Otimização de linha de costura por graph-cut (o que ferramentas
     comerciais maduras usam para evitar misturar através de objetos em
     movimento/erros de paralaxe) não está implementada.
  4. Escreve um GeoTIFF **RGBA** (4 bandas, uint8) via `rasterio`: pixels
     sem nenhuma câmera contribuinte ficam com alpha=0 (NoData), nunca com
     uma cor "inventada" ou escondidos.

### Achado da Fase 6: bug de convenção de câmera na cena sintética de teste

A validação da Fase 6 expôs um bug real, pré-existente, na fixture de
teste compartilhada (`tests/synthetic_scene.py`), não no código de
produção. O renderizador `_render_nadir_crop` usava um sinal de mapeamento
linha/coluna que produzia imagens espelhadas norte-sul em relação a uma
câmera pinhole física real. Isso nunca afetou os testes das Fases 2-5
porque eles só validam **posição** do centro de câmera após SfM +
alinhamento por similaridade ao GNSS (uma operação insensível a essa
convenção de orientação/mão), nunca a orientação absoluta.

O bug só apareceu ao construir, para a Fase 6, uma reconstrução de
"verdade de campo" manualmente (via
`add_camera_with_trivial_rig`/`add_image_with_trivial_frame`/
`Frame.set_cam_from_world`/`Reconstruction.register_frame`) — necessário
porque SfM real sobre a cena sintética plana, nadir-only e de altitude
constante do projeto é uma configuração genuinamente degenerada para a
recuperação de **pose absoluta** (achado distinto da degenerescência de
autocalibração de intrínsecos já documentada na Fase 3, mas com a mesma
raiz: "plano + só-nadir + altitude constante"). Ao derivar a rotação
correta para essa reconstrução manual (regra da mão direita: câmera
right-handed com X=leste e Z=direção-de-visada-para-baixo implica
Y=**sul**, não norte — logo uma câmera nadir fisicamente válida mostra
norte no topo da imagem), ficou evidente que o renderizador da fixture
não seguia essa convenção. Usar a rotação fisicamente correta
(`diag(1, -1, -1)`, determinante +1) contra o renderizador com o sinal
antigo produzia uma matriz de rotação **inválida** (reflexão, determinante
-1) quando "corrigida" para combinar com o bug do renderizador —
`pycolmap.Rotation3d` aceita silenciosamente uma matriz de reflexão e
produz um quaternion não-normalizado corrompido, sem lançar erro,
corrompendo toda a geometria a jusante (`projection_center()` com sinal
errado, projeções "válidas" que na verdade eram lixo). Corrigido em
ambos os lugares: o sinal do renderizador (`_render_nadir_crop`) e a
rotação da reconstrução manual, restaurando a convenção fisicamente
correta em toda a fixture. Nenhum teste das Fases 2-5 foi afetado pela
correção (suíte completa re-executada e confirmada sem regressão).

### Validação com verdade de campo conhecida

Cena sintética com posições de câmera **exatas** e conhecidas (sem
depender de SfM, pelo motivo de degenerescência acima), terreno de
textura suave (baixa frequência espacial) para isolar corretude
geométrica de ruído de sub-pixel, e DEM plano sintético com margem além
da cobertura das câmeras. Validado: cor reconstruída no ortomosaico
contra a cor **verdadeira** conhecida do terreno em dezenas de posições
aleatórias dentro da área coberta (erro médio < 8 em valor de canal de
cor, 0-255); pixels fora de toda cobertura de câmera corretamente
marcados NoData (alpha=0); rejeição de configuração inválida
(`feather_fraction` fora de (0, 0.5]); erros claros quando reconstrução
ou DEM de entrada não existem.

- `core.project`: novo `OrthoSummary` (raster_path, dimensões, resolução,
  câmeras usadas, pixels válidos/NoData).
- `core.report`: quando `project.ortho` existe, preenche a seção
  "Orthomosaic" (tamanho, sistema de coordenadas) e o campo Orthomosaic
  dos Processing Parameters.
- CLI: `htrmapper ortho <projeto.json> --output <ortho.tif> [--feather-fraction N]`
  (requer Fase 4 e Fase 5 já executadas). GUI: botão "Gerar Ortomosaico…".

## 18. Fase 7 — interface completa (implementada e validada)

Substitui a janela de página única das Fases 1-6 por uma árvore de projeto
(`Projeto / Imagens / Câmeras / Tie Points / Point Cloud / DEM /
Orthomosaic`, exatamente como pedido no briefing original), com três
capacidades novas que o briefing também pede explicitamente: progresso das
etapas de longa duração, cancelamento, e monitoramento de CPU/RAM/GPU/VRAM.

### Execução em segundo plano (`gui.worker.PipelineWorker`)

Toda etapa de longa duração (Alinhar, Ajustar, Nuvem densa, DEM,
Ortomosaico) passa a rodar numa `QThread` dedicada em vez de bloquear o
event loop do Qt -- a árvore, os botões e o monitor de hardware continuam
respondendo durante todo o processamento. `PipelineWorker` inspeciona a
assinatura da função-alvo (`inspect.signature`) para decidir, por etapa,
se ela aceita `cancellation_token`/`progress_callback`, em vez de assumir
que toda função aceita -- a Fase 3 (ajuste GNSS) não declara esses
parâmetros porque o `Ceres::Solve` usado por
`pycolmap.create_pose_prior_bundle_adjuster` não expõe nenhum gancho de
cancelamento nesta versão do pycolmap, então a GUI simplesmente não mostra
um botão Cancelar para essa etapa -- nunca um botão decorativo que não
faz nada.

### Cancelamento real (`pycolmap.CancellationToken`)

Cada função de pipeline que o COLMAP já suporta nativamente
(`extract_features`, `match_spatial`/`match_exhaustive`,
`incremental_mapping`, `undistort_images`, `patch_match_stereo`,
`stereo_fusion`) passou a aceitar um parâmetro opcional
`cancellation_token: pycolmap.CancellationToken`, repassado diretamente à
chamada pycolmap correspondente -- confirmado empiricamente (não apenas
assumido pela documentação) que um token já cancelado faz essas chamadas
levantarem `InterruptedError` imediatamente. O laço de reprojeção por
câmera do Ortomosaico (Fase 6, código próprio, não COLMAP) reaproveita o
mesmo `CancellationToken` como uma flag thread-safe barata, checando
`.is_cancelled` a cada câmera. A geração de DEM (Fase 5, numpy/scipy puro,
sem chamada COLMAP nenhuma) aceita qualquer objeto com uma propriedade
`is_cancelled` (duck typing) e checa entre as fases (carregar, filtrar,
interpolar, escrever) -- cancelamento apenas nas fronteiras de fase, nunca
fabricado como granularidade fina que a computação não tem. Em todos os
casos, cancelado levanta `InterruptedError`, nunca mascarado como um erro
genérico de pipeline -- é assim que o worker distingui "cancelado" de
"falhou" (`worker.cancelled` vs. `worker.failed`).

### Progresso real, não fabricado

`progress_callback` segue o mesmo princípio: para as etapas apoiadas no
COLMAP, só progresso por fase (rótulos como "Extraindo features (SIFT)",
"Reconstrução incremental (SfM)") é reportado, porque a API de alto nível
do COLMAP não expõe percentual de conclusão dentro dessas chamadas -- uma
barra de progresso indeterminada (spinner) é o que a GUI mostra para essas
etapas, nunca uma porcentagem inventada. Para o Ortomosaico (laço próprio,
por câmera), progresso real e granular (`câmeras_processadas/total`) é
reportado a cada câmera -- uma barra de progresso determinada de verdade,
porque aqui o código realmente sabe quanto falta.

### Monitor de CPU/RAM/GPU/VRAM ao vivo (`core.system_info.sample_live_usage`)

Amostra `psutil.cpu_percent()`/`psutil.virtual_memory()` (leitura direta
de `/proc`, sem subprocess) e `nvidia-smi --query-gpu=...` (utilização e
VRAM) a cada 1 segundo via `QTimer`. Sem GPU NVIDIA, mostra "sem GPU
NVIDIA" -- nunca "0% de uso", que confundiria "não medido" com "medido,
ocioso" (mesmo princípio de nunca fabricar um valor para o que não foi
calculado, já usado no `core.report`).

### Achado da Fase 7: nunca chamar subprocess durante uma etapa de pipeline em segundo plano

Testando o monitor de hardware junto com um worker de Alinhamento (SfM)
rodando em segundo plano, uma suspensão real e reproduzível apareceu: o
monitor chamando `nvidia-smi` (via `subprocess.run`, que no Linux usa
`fork()`) exatamente enquanto o COLMAP tinha várias threads nativas ativas
(extração SIFT paralela) travou o processo inteiro. Esse é o clássico
problema de `fork()` num processo multi-thread: um lock que outra thread
segurava no instante do fork é copiado para o processo filho já "travado",
mas a thread que o liberaria não existe mais lá -- o filho (e quem espera
por ele) pode travar para sempre. A correção: `sample_cpu_ram_usage()`
(sem nenhum subprocess) é usada enquanto qualquer worker de pipeline está
ativo neste processo (contador global, não por-janela, já que `fork()`
afeta o processo inteiro); a amostragem de GPU via `nvidia-smi` só volta a
rodar quando nenhum worker está ativo. Confirmado que essa suspensão
específica não era o problema real de um segundo achado nesta mesma
investigação -- ver abaixo -- mas é uma correção real e válida por conta
própria, mantida por ser engenharia sólida (nunca fazer fork de um
processo pesadamente multi-thread por uma amostragem periódica opcional).

### Achado da Fase 7: ordem de atualização de estado após um worker terminar

A investigação acima revelou uma trava real diferente: `QMessageBox.exec()`
(o diálogo modal "Alinhamento concluído") bloqueia esperando um clique que
nunca chega num teste automatizado/headless -- não é um bug (um usuário
real clica normalmente), apenas algo que todo teste de UI Qt precisa
simular (o teste correspondente faz `monkeypatch` de `QMessageBox.exec`
para aceitar automaticamente). Investigar esse bloqueio, porém, expôs um
bug real: `_end_worker()` chamava `_sync_ui_to_project_state()` (que
decide todo botão habilitado/desabilitado e todo painel atualizado) ANTES
de `on_success(result)` gravar o resultado em `self.project` -- então
"Ajustar (GNSS)" continuava desabilitado mesmo depois do Alinhamento (SfM)
terminar com sucesso, porque a sincronização da UI rodava contra o estado
antigo do projeto. Corrigido invertendo a ordem: `on_success` grava o
resultado primeiro, só then a sincronização final da UI roda. Um teste
(`test_align_worker_updates_project_and_ui`) prova especificamente que o
botão "Ajustar (GNSS)" fica habilitado após um Alinhamento real terminar,
não apenas que a chamada não lança exceção.

### Validação

`tests/test_cancellation_and_progress.py`: um token já cancelado levanta
`InterruptedError` real (não simulado) para SfM, Ortomosaico e DEM;
callbacks de progresso disparam nas fases/contagens esperadas, em ordem.
`tests/test_gui_main_window.py`: roda com o plugin de plataforma Qt
"offscreen" (sem display real necessário); constrói a janela, navega pela
árvore, importa imagens sintéticas, executa um Alinhamento (SfM) real em
segundo plano até o fim via um event loop Qt local (não um mock), e
confirma que `self.project.sfm` foi preenchido e os botões corretos foram
habilitados -- não apenas "não lançou exceção". Também confirma que o
Ajuste (GNSS) não oferece Cancelar (sem gancho de cancelamento nativo) e
que os painéis de DEM/Ortomosaico renderizam uma prévia raster real sem
travar.

- `core.system_info`: `sample_live_usage()`/`sample_cpu_ram_usage()`
  (Fase 7, monitor ao vivo) somam-se a `detect_system_info()` (Fase 1,
  descrição estática de hardware para o relatório).
- GUI: `gui/worker.py` (novo), `gui/main_window.py` (reescrito).

## 19. Revisão de consistência das Fases 1-7

Revisão pedida explicitamente pelo usuário após a Fase 7, comparando
código, testes e documentação das sete fases entre si (não uma fase nova).
Achados reais e correções aplicadas:

1. **Numeração de seções deste documento**: pulava de "12" direto para
   "14" (seção "13" nunca existiu). Conteúdo estava completo, só a
   numeração estava errada -- corrigido renumerando 14→13 até 19→18 e
   todas as referências cruzadas (neste arquivo e no `README.md`), exceto
   a única referência a "seção 17 do briefing original" (documento externo
   do usuário, não deste arquivo -- não renumerada).
2. **`core.report`, seção "Configuração de Precisão GNSS/PPK"**: o texto
   era um parágrafo fixo, sempre igual, dizendo "Nenhum ajuste foi
   executado ainda" -- mesmo depois da Fase 3 já ter rodado de verdade e a
   seção "Camera Locations" logo acima já mostrar o RMSE real do ajuste.
   Corrigido para checar `camera_calibration.pending_phase` (o mesmo sinal
   que a seção "Camera Calibration" já usa) e mostrar o texto certo em
   cada caso.
3. **`core.report`, parágrafos de "Processing Parameters"**: os parágrafos
   de Tie Points / Depth Maps / Point Cloud / DEM / Orthomosaic usavam a
   classe CSS `pending` (âmbar, itálico -- reservada para "não calculado
   ainda", ver `theme.PENDING`) incondicionalmente, mesmo depois da fase
   correspondente já ter rodado e o parágrafo mostrar um valor real
   calculado. Corrigido com um helper (`_metric_paragraph`) que escolhe a
   classe CSS pelo próprio `Metric.is_available`.
4. **`core.report`, `CameraCalibration.note`**: o texto padrão dizia
   "ainda não implementada" (bundle adjustment não implementado) quando na
   verdade a Fase 3 já está implementada há muito tempo -- só não rodou
   ainda *para aquele projeto*. Corrigido para "ainda não executada",
   consistente com a linguagem usada em todo o resto do relatório (ex.
   `aligned_cameras`: "Alinhamento (SfM) ainda não executado").
5. **`core.project.ProjectCrsConfig.export_epsg`**: campo existia desde a
   Fase 1, com um comentário prometendo "defaults to project_epsg if
   None" -- mas nenhum caminho de exportação (LAS na Fase 4, GeoTIFF do
   DEM na Fase 5) de fato o lia; todos usavam `project_epsg` diretamente,
   então configurar um `export_epsg` diferente não tinha efeito nenhum,
   silenciosamente. Corrigido adicionando a property
   `effective_export_epsg` (`export_epsg` quando definido, senão
   `project_epsg`) e trocando os pontos de chamada em `mvs.dense`, `cli`
   e `gui` para usá-la em vez de `project_epsg` bruto.
6. **`mvs.dense.MvsResult`**: a divisão de `undistorted_image_path` em
   duas variáveis (`undistorted_image_path` = pasta `images/`,
   `undistorted_reconstruction_path` = pasta `sparse/`), feita durante a
   Fase 6/7 para a Fase 6 conseguir localizar seus insumos, nunca tinha um
   teste confirmando que os caminhos reais em disco batem com o que
   `pycolmap.undistort_images` de fato escreve. Adicionado um teste que
   roda a desdistorção de verdade (sem GPU real, via
   `monkeypatch.setattr(pycolmap, "has_cuda", True)` só para passar da
   checagem inicial) e confirma as duas subpastas e seus conteúdos.
7. **`core.report`, tabela de câmeras**: dentro do agrupamento por
   `camera_model`, duas variáveis (`widths`/`heights`, os conjuntos de
   larguras/alturas distintas no grupo) eram calculadas mas nunca lidas --
   sinal de uma checagem que ficou incompleta. A resolução mostrada na
   tabela vinha sempre da PRIMEIRA imagem do grupo, mesmo quando outras
   imagens com o mesmo nome de `camera_model` tinham dimensões de pixel
   diferentes (um problema real de qualidade de dados -- sensores
   diferentes reportados sob o mesmo nome de modelo, ou um bug de
   agrupamento -- escondido silenciosamente). Corrigido: quando o grupo
   tem mais de uma largura ou altura distinta, a tabela mostra "MISTA (N
   larguras, M alturas distintas)" em vez de um valor arbitrário.

Nenhuma inconsistência encontrada nas fases 2-6 quanto à precisão
geométrica/GNSS em si (pesos, CRS de trabalho, convenção de câmera) --
essas já tinham sido validadas com dados sintéticos de verdade de campo
conhecida em cada fase própria. Suíte completa (agora com os testes de
regressão acima) re-executada sem regressão.

## 20. Achados da primeira execução real (56 fotos reais, 21MP, Mavic 3M)

A primeira execução do pipeline com dados reais do usuário (não mais
sintéticos) expôs dois problemas que nenhum teste com imagens sintéticas
pequenas (320x240) poderia revelar -- ambos corrigidos:

1. **Eixos do gráfico de posições de câmera ilegíveis**: coordenadas UTM
   (6-7 dígitos) faziam o matplotlib mostrar um offset pequeno
   ("+7.548e6") com marcações relativas curtas embaixo -- a coordenada
   real nunca aparecia diretamente. Corrigido em `gui.main_window._plot_points`
   e `core.report._render_camera_position_map_png` (o mesmo gráfico
   aparece nos dois lugares) para mostrar o número completo em cada
   marcação, sem notação científica, giradas na vertical para não ocupar
   espaço horizontal excessivo.
2. **Alinhamento (SfM) extremamente lento em fotos reais**: no relato do
   usuário, o matching de um único par de imagens chegou a levar mais de
   5 minutos, com fotos individuais reportando 60000-75000 features SIFT
   -- bem acima do limite de 40000 configurado. Investigação e correção
   completas:
   - `sfm.pipeline.SfmConfig` nunca configurava
     `FeatureExtractionOptions.max_image_size` (o padrão do próprio
     pycolmap é -1, sem redução de resolução) -- então imagens reais de
     ~21MP (5280x3956) eram processadas em resolução nativa. O próprio log
     do COLMAP já avisa disso ("Consider reducing the maximum image
     size...").
   - **Achado mais sutil, verificado empiricamente antes de corrigir**: o
     extrator SIFT em CPU do COLMAP só de fato reamostra a imagem quando o
     tamanho pedido cruza uma fronteira de "oitava" da pirâmide interna
     (passos de potência de 2 a partir da resolução nativa). Testado
     diretamente em imagem de 5280px de largura: `max_image_size=3200`
     (redução de ~1.65x) mediu tempo e contagem de features **idênticos**
     a nenhum limite -- um no-op silencioso. Só a partir de reduções que
     cruzam essa fronteira (`max_image_size=2000`, testado e confirmado)
     o tempo de extração caiu de fato (mais da metade, no teste). Por
     isso o padrão escolhido é 2000, não um valor "redondo" como 3200
     (que aliás é o mesmo já usado na Fase 4 para a nuvem densa, onde essa
     mesma armadilha não se aplica da mesma forma).
   - `max_num_features` também se confirmou um limite aproximado, não
     rígido: em teste controlado, configurar 40000 ainda retornou ~43000
     features numa imagem de resolução real rica em textura -- o COLMAP
     distribui o orçamento por oitava e pode passar do total.
   - CLI: nova flag `htrmapper align --max-image-size` (padrão 2000),
     simétrica a `--key-point-limit`. GUI usa o mesmo padrão.
   - Teste de regressão (`tests/test_sfm_pipeline.py`) mede **tempo**, não
     contagem de features, numa imagem sintética nas dimensões reais
     (5280x3956) -- contagem de features em ruído aleatório se mostrou não
     monotônica em função do tamanho (ruído é um caso adversarial para um
     detector em espaço de escala), então o teste original baseado nisso
     foi descartado por ser ele mesmo não-confiável, não só o código que
     testava.

## 21. "Tie Point Limit" (Metashape) -- pesquisado, decidido não implementar agora

O usuário pediu explicitamente para pesquisar como Agisoft Metashape e
Pix4Dmapper lidam com desempenho de matching, e se valia a pena
implementar o equivalente ao "Tie Point Limit" do Metashape. Pesquisa
feita (documentação/comunidade das duas ferramentas) e testada
empiricamente contra o próprio COLMAP antes de decidir -- não só lida
como consenso de manual.

**O que "Tie Point Limit" realmente é**: um corte que acontece *depois*
do matching entre pares de imagem -- quantos pontos correspondidos ficam
guardados por imagem, não quantos são calculados. É diferente do "Key
Point Limit" (que já ajustamos, esse sim antes do matching). O
equivalente direto no COLMAP é
`pycolmap.FeatureMatchingOptions.max_num_matches` (padrão 32768).

**Testado diretamente, não assumido**: matching de duas imagens sintéticas
nas dimensões reais de drone (5280x3956, ~43000 features cada, perto do
que fotos reais do usuário geram) variando esse limite:

```
max_num_matches=32768: 4.07s
max_num_matches=8000:  4.02s
max_num_matches=2000:  4.07s
```

**Zero diferença de tempo.** Faz sentido: o corte acontece depois da
comparação de descritores (força bruta, N×M) já ter sido feita -- reduzir
quantos pontos ficam guardados no final não evita o trabalho pesado que
já rodou. Implementar isso agora não atacaria o problema real do usuário
(alinhamento lento).

**Onde o tempo real estava indo**: no log real do usuário, a primeira
imagem levou 339s contra suas vizinhas espaciais, a segunda só 73s --
não é o tempo de UM par, é o tempo de casar uma imagem contra *todos* os
candidatos dentro do raio de busca (`spatial_max_distance_m=150m`,
`spatial_max_neighbors`, antes 50). Contra ~7s por par (medido acima em
condição parecida), 339s ÷ 7s ≈ 48 pares -- bate com o limite de 50
vizinhos sendo atingido. Isso é exatamente o que Metashape (Generic/
Reference preselection) e Pix4D (Aerial Grid or Corridor) fazem de forma
mais inteligente: restringem os PARES candidatos usando a geometria do
voo, não um raio genérico.

**Decisão**: reduzido `spatial_max_neighbors` de 50 para **30** --
confirmado pela própria FAQ do COLMAP ("to reduce matching pairs, decrease
SpatialMatching.max_num_neighbors") e por exemplos publicados de pipelines
reais usando esse mesmo valor para datasets aéreos grandes. Exposto na
CLI (`htrmapper align --spatial-max-neighbors`) e na GUI (segundo campo no
mesmo diálogo do limite de features). "Tie Point Limit" propriamente dito
fica anotado como candidato à Fase 9 (controle de custo do bundle
adjustment em datasets muito grandes, centenas de imagens), não como
correção de desempenho agora.

### Faixas recomendadas (GUI e `--help` da CLI)

Nem inventadas, nem escolhidas por tentativa e erro -- vêm da orientação
publicada do próprio Metashape (key point limit) e da FAQ/exemplos
publicados do COLMAP (spatial max neighbors), citadas acima.

| Perfil | Key point limit | Spatial max neighbors | Quando usar |
|---|---|---|---|
| Rápido | 8.000 | 15 | Voo em grade regular, boa sobreposição, GNSS confiável, prioridade em velocidade |
| Padrão | 15.000 | 30 | Uso geral -- padrão atual da GUI |
| Preciso / textura densa | 40.000+ | 50 | Vegetação/dossel/cultivo denso (como o caso do usuário), voo irregular, ou GNSS menos confiável -- prioridade em redundância, não em velocidade |

## 22. GPU real não detectada mesmo com hardware NVIDIA presente

O usuário relatou uma RTX 3060 Ti, mas a Fase 4 (nuvem densa) falhou com
"nenhuma GPU NVIDIA detectada" -- e o próprio log da Fase 2 já mostrava
"Creating SIFT CPU feature extractor"/"matcher" o tempo todo, nunca GPU.
Pesquisado antes de assumir que era um bug de detecção: **não é** -- é uma
limitação real e documentada do pacote instalado.

O `pip install pycolmap` no Windows (e em qualquer plataforma via o pacote
`pycolmap` "normal" do PyPI) **não inclui suporte a CUDA nenhum** -- nem
para SIFT em GPU, nem para o MVS (patch-match stereo) que a Fase 4 precisa.
Suporte a CUDA só existe hoje via um pacote separado, `pycolmap-cuda12`,
e só para Linux. Não existe wheel com CUDA para Windows no PyPI atualmente
-- a única forma de ter CUDA nativo no Windows é compilar o pycolmap do
zero (toolchain vcpkg + CUDA Toolkit + Visual Studio), um processo bem
mais trabalhoso.

**Caminho prático recomendado, se o usuário quiser usar a RTX 3060 Ti**:
WSL2 (Windows Subsystem for Linux) com passthrough de GPU NVIDIA -- esse é
um recurso oficial e maduro da própria NVIDIA/Microsoft (drivers Windows
já bastam, não se instala driver Linux dentro do WSL2), com desempenho
próximo do nativo. Dentro do WSL2 (Ubuntu, por exemplo), `pip install
pycolmap-cuda12` funcionaria de verdade com a GPU, para SIFT em GPU e para
a nuvem densa da Fase 4. Isso não foi implementado nem testado por nós
(depende do ambiente do próprio usuário), só documentado como o caminho
correto -- nunca prometemos que funcionaria sem essa mudança de ambiente.
