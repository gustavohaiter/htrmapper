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
Fase 2) convergiu de forma limpa e estável. Por isso os intrínsecos ficaram
fixos por padrão até o primeiro voo real mostrar o custo disso: ver seção
27. Hoje são refinados por padrão, com travas contra exatamente essa
degenerescência.

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

**O ambiente de desenvolvimento usado neste projeto não possui GPU** — a
etapa de cálculo denso em si não pôde ser exercitada diretamente aqui.
Isso foi dito explicitamente enquanto era verdade, para não alegar uma
validação que não tinha ocorrido (regra 3 do briefing).

**Atualização: validado com hardware real em 2026-10-06/07**, pelo
usuário, numa RTX 3060 Ti via WSL2 (ver seção 23) — 56 fotos reais do
DJI Mavic 3M, qualidade "alta" (`max_image_size=3200`): patch-match
stereo levou 84 minutos, fusão ~1 minuto, resultado em 90.246 pontos
fundidos exportados para LAS, DEM e ortomosaico gerados em seguida sem
erro. Essa execução real também expôs um bug genuíno (não hipotético) no
módulo (ver "Bug real encontrado..." logo abaixo) — corrigido e
confirmado funcionando no mesmo hardware, via um script de recuperação
que reaproveitou os depth maps já calculados em vez de refazer os 84
minutos de GPU.

O fallback de CPU para máquinas sem GPU continua não implementado — não
é mais um requisito urgente agora que o caminho GPU via WSL2 está
validado, mas segue reservado no desenho (OpenMVS externo, seção 2) caso
surja necessidade de processar numa máquina sem GPU.

### Bug real encontrado na primeira execução com GPU de verdade (corrigido)

`pycolmap.stereo_fusion()` recebe um parâmetro `output_type` (padrão
`"bin"`) que controla como o `Reconstruction` retornado é lido de volta
depois da fusão. Sem especificar `output_type="PLY"` explicitamente, o
binding do `pycolmap-cuda12` 4.2.1 tenta ler o `output_path` dado (um
arquivo `.ply`) como se fosse um diretório de reconstrução binária do
COLMAP (`cameras.bin`/`images.bin`/`points3D.bin`), e falha com
`ValueError: Check failed: colmap::ExistsDir(path_val)` — mesmo com a
fusão em si já tendo concluído com sucesso e o `fused.ply` já escrito
corretamente em disco pelo núcleo em C++ antes do binding Python falhar
no pós-processamento. Esse bug só podia ser encontrado rodando de
verdade (nunca apareceria nos testes sintéticos, que não exercitam
`stereo_fusion` por não terem GPU) — exatamente o tipo de achado que a
regra "nunca confiar em validação não testada" deste projeto existe
para capturar. Corrigido em `mvs/dense.py` passando `output_type="PLY"`
explicitamente.

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
  2. **Resolução automática** (hoje 2,0x, ver seção 28): `espaçamento_médio × 2,5`, onde
     `espaçamento_médio = sqrt(área / número_de_pontos)`. Heurística
     documentada como tal (não uma lei física): uma célula do DEM precisa
     de suporte de múltiplos pontos da nuvem densa para interpolar de
     forma confiável, e a prática comum em fotogrametria aérea recomenda
     resolução de DEM/ortomosaico de 2-4x o espaçamento de pontos/GSD.
     Sempre sobrescrevível pelo usuário (`DemConfig.resolution_m`).
  3. **Interpolação + preenchimento de buracos**: substituídos na seção
     28 (mediana por célula, filtro de picos, preenchimento só dentro da
     área reconstruída). O método original abaixo (interpolação linear
     dos pontos crus + nearest-neighbor até o retângulo envolvente)
     deixava picos de metros e inventava terreno fora dos dados.
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
  3. **Blending por distância-à-borda ("feathering")** — substituído na
     seção 26 pela escolha da foto mais vertical por pixel: cada câmera
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
- CLI: `htrmapper ortho <projeto.json> --output <ortho.tif> [--resolution N] [--max-view-angle G]`
  (`--feather-fraction` existiu até a seção 26, que trocou a mistura por média)
  (requer Fase 4 e Fase 5 já executadas). GUI: botão "Gerar Ortomosaico…".

### Bug real encontrado com dados reais em 2026-10-07: resolução do ortomosaico acoplada à do DEM

A primeira execução de ponta a ponta com fotos reais (fazenda 20046, 56
fotos DJI Mavic 3M) produziu um ortomosaico **visivelmente ruim**
("blocado"/pixelado) — o usuário comparou com a experiência diária dele
em Pix4D/Metashape e corretamente desconfiou do resultado antes de aceitar
qualquer coisa como "deu certo". Investigação confirmou: o ortomosaico
saiu em **2,46 m/pixel**, quando a resolução nativa das fotos nesse voo
(~100m AGL) é de **~2,7 cm/pixel** — quase 100x mais grosseiro. Comparação
direta com metadados de um export real do Pix4Dfields do mesmo voo
confirmou o alvo correto (0,0248 m/pixel, 13756x12215px) e também revelou
que a área realmente fotografada no solo (~341m x ~303m) é bem maior que
a extensão apenas das *posições das câmeras* (~201m x ~161m, o número que
o relatório de importação mostra) — a pegada de cada foto no solo se
estende bem além de onde o drone passou.

**Causa raiz**: `run_orthomosaic_generation` carregava a grade de saída
**diretamente do raster do DEM** (`_load_dem_grid`), pixel a pixel —
então qualquer resolução que o DEM tivesse (automática, derivada da
densidade da nuvem de pontos — baixa nesse voo, ver seção 24 abaixo) virava,
sem nenhum desacoplamento, a resolução do ortomosaico também. Isso é uma
diferença de desenho real em relação a ferramentas maduras: no
Pix4D/Metashape, o DEM/malha é usado **só** como consulta de altura de
terreno na hora de orto-retificar; a resolução do ortomosaico em si é
escolhida de forma independente, perto da resolução nativa das fotos.

**Correção**: `run_orthomosaic_generation` agora constrói sua **própria**
grade de saída (mesma extensão/footprint do DEM, resolução independente)
e reamostra a superfície de elevação do DEM sobre essa grade nova via
interpolação bilinear (`scipy.interpolate.RegularGridInterpolator` — nunca
reamostragem própria). A resolução automática (quando não especificada) é
estimada diretamente da altura real de cada câmera registrada acima do
terreno do DEM e da distância focal calibrada em pixels
(`Camera.mean_focal_length()`) — a mesma fórmula pinhole de `geo.gsd`, mas
usando geometria reconstruída de verdade em vez da altitude nominal do
EXIF, que é mais precisa (exatamente a recomputação "pós-alinhamento,
mais precisa" que o próprio docstring de `geo/gsd.py` já antecipava desde
a Fase 1, nunca implementada até agora). Validado: teste de regressão
confirma que a resolução automática do ortomosaico não muda quando o DEM
de entrada é 10x mais grosseiro; outro teste confirma que a resolução
automática bate com o GSD conhecido da cena sintética (tolerância 10%).

`OrthoConfig` ganhou `resolution_m: float | None = None` (mesmo padrão do
`DemConfig`); `OrthoResult`/`OrthoSummary` ganharam `resolution_source`
("automatic (camera GSD)" / "user-defined" / fallback para a resolução do
DEM no caso degenerado de nenhuma câmera ter amostra de terreno válida).
CLI: nova flag `--resolution`. GUI: novo diálogo de resolução no botão
"Gerar Ortomosaico…", espelhando o que já existia para o DEM.

**Ver seção 24** para a investigação de por que a nuvem densa desse voo
real saiu com densidade baixa (~1-3 pontos/m², quando MVS tipicamente
entrega dezenas a centenas).

### Bug real encontrado no mesmo dia (2026-10-07), consequência direta do fix acima: sem memória (OOM) na resolução nativa

Corrigir o acoplamento de resolução (acima) teve uma consequência real não
antecipada: numa área fotografada real de ~341m x ~303m a ~2,7cm/px, a
grade de saída do ortomosaico passou a ter **~168 milhões de pixels**
(contra ~14 mil pixels do bug antigo). A implementação de então montava
várias matrizes `float64` do tamanho da grade **inteira** simultaneamente
(cor acumulada, peso, coordenadas de mundo, elevação) — e o processo foi
morto pelo OOM killer do Linux ao tentar rodar isso de verdade no PC do
usuário. Esse é exatamente o tipo de problema de escala que só aparece
processando dados reais de tamanho real — nunca apareceria nas cenas
sintéticas de teste (pequenas de propósito).

**Correção**: `run_orthomosaic_generation` agora processa o raster de
saída **em blocos/tiles** (`TILE_SIZE_PX = 2048`, nunca exposto como
configuração do usuário por enquanto), a mesma técnica que Pix4D/
Metashape usam — nunca mantêm o ortomosaico inteiro em RAM de uma vez.
Para cada tile: calcula só as coordenadas/elevação desse pedaço,
acumula cor/peso só nesse pedaço, escreve via `rasterio` com janela
(`Window`) direto no GeoTIFF, descarta tudo antes do próximo tile. Pico
de memória fica limitado ao tamanho do tile, nunca ao tamanho total da
saída.

Para não testar as 56 câmeras contra cada um dos ~42 tiles sem
necessidade, cada câmera registrada tem sua pegada aproximada no solo
pré-calculada **uma vez**, reprojetando a grade do DEM (pequena, nativa)
através dela — reaproveitando a mesma matemática de projeção do loop
principal, só que sobre poucos pontos em vez de milhões — e usada como
filtro barato de bounding-box por tile (o teste preciso pixel a pixel só
roda para as câmeras cujo bounding-box realmente sobrepõe o tile).

**Validado**: um teste de regressão roda a mesma cena sintética duas
vezes — uma com `TILE_SIZE_PX` forçado para um valor gigante (um único
tile, equivalente ao comportamento antigo) e outra forçado para 8
(centenas de tiles minúsculos) — e confirma que o GeoTIFF resultante é
**idêntico byte a byte** nos dois casos. Isso prova que o tiling é só um
detalhe de implementação/memória, nunca muda o resultado nem introduz
costuras nas bordas dos blocos.

**Limite ainda conhecido, não resolvido**: as imagens-fonte decodificadas
(câmera → array RGB) ficam em cache em memória pelo resto da execução,
assim que uma câmera é usada pela primeira vez — correto e barato para
um voo de dezenas de fotos (ex. os 56 da fazenda 20046, ~3,5GB no pior
caso), mas cresceria sem limite para um voo de centenas/milhares de fotos
de alta resolução. Não implementado ainda (um cache com despejo
tipo LRU resolveria) porque não é o cenário de uso atual do usuário
(talhões pequenos/médios de cana) — documentado aqui para não ser
esquecido se um voo bem maior aparecer no futuro.

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
a nuvem densa da Fase 4.

**Atualização: validado em 2026-10-06/07** pelo próprio usuário, numa
RTX 3060 Ti via WSL2/Ubuntu 26.04 -- `pycolmap-cuda12` reconheceu a GPU
(`has_cuda: True`) sem precisar instalar nenhum driver Linux nem CUDA
Toolkit manualmente no sistema (o pacote já traz seu próprio runtime
CUDA via pip, como `nvidia-cuda-runtime-cu12`); a Fase 4 completa (56
fotos reais) rodou com sucesso -- ver seção 15 para os números e o bug
real encontrado/corrigido nessa mesma execução.

## 23. Workflow Windows + WSL2 (validado com hardware real em 2026-10-06/07)

Decisão de arquitetura: em vez de o HTRMapper "chamar o WSL2 por dentro"
(subprocess/RPC entre um processo Windows e um processo WSL2 -- complexo,
e não testável neste ambiente de desenvolvimento, que não tem WSL2 real),
o desenho aproveita o que o próprio WSL2 já resolve nativamente: ele
enxerga o disco do Windows em `/mnt/<letra>/...`. O fluxo vira:

1. Fases 1-3 (Importar, Alinhar, Ajustar) na GUI do Windows, normalmente.
2. Fases 4-6 (`htrmapper dense`/`dem`/`ortho`) rodadas de um terminal
   dentro do WSL2, com um ambiente Python que tem `pycolmap-cuda12`,
   apontando para o **mesmo** `projeto.json`.

**Problema técnico resolvido**: o `projeto.json` salvo pelo Windows grava
caminhos como `C:\Users\...`, que não existem dentro do WSL2 (lá é
`/mnt/c/Users/...`). Sem tratar isso, o comando de dentro do WSL2 não
acharia as imagens nem a reconstrução salva antes.

- `core.platform_paths`: detecta se o processo atual está rodando dentro
  do WSL2 (`WSL_DISTRO_NAME` no ambiente, ou a string "microsoft" em
  `/proc/version` -- os dois sinais padrão de detecção de WSL, nenhum
  depende de chamar uma ferramenta externa) e traduz qualquer caminho
  Windows (`C:\...`) para o equivalente WSL2 (`/mnt/c/...`), e o inverso
  quando rodando nativamente no Windows. Puro tratamento de string,
  testável sem WSL2 de verdade (o que só o próprio WSL2 real garante --
  que o mount `/mnt/<letra>` de fato se comporta como a NVIDIA/Microsoft
  documentam -- não é responsabilidade deste módulo nem foi validado
  aqui).
- Aplicado em todo `from_dict()` de `core.project` que carrega um campo de
  caminho (`ImageRecord.path`, e os caminhos de banco/reconstrução/nuvem/
  raster de cada summary de fase) -- então um `projeto.json` salvo no
  Windows continua funcionando quando `htrmapper dense/dem/ortho` roda de
  dentro do WSL2, sem o usuário precisar editar o JSON à mão ou
  reimportar as imagens.
- Mensagem de erro da Fase 4 (`MvsError`, quando não há CUDA) atualizada
  para apontar esse caminho explicitamente, em vez de só dizer "sem GPU".

### Setup do WSL2 (validado passo a passo pelo usuário em 2026-10-06)

Nada disso pôde ser executado neste ambiente de desenvolvimento (é um
container Linux comum, sem WSL2) -- documentado originalmente como o
caminho correto a partir da documentação oficial NVIDIA/COLMAP, sem
alegar validação que não tinha ocorrido. O usuário então seguiu esse
roteiro na própria máquina (PC pessoal, Ryzen 7 5700G + RTX 3060 Ti,
placa-mãe Gigabyte B550M Aorus Elite) e encontrou um obstáculo real não
previsto aqui: a virtualização (SVM, equivalente AMD do VT-x) vinha
desabilitada de fábrica na BIOS, causando
`HCS_E_HYPERV_NOT_INSTALLED` ao instalar o WSL2 mesmo após habilitar o
componente "Plataforma de Máquina Virtual" do Windows -- resolvido
habilitando `SVM Mode` em
`BIOS → Tweaker → Advanced CPU Settings → SVM Mode`. Esse é exatamente
o tipo de obstáculo específico de hardware que uma documentação escrita
sem testar de verdade não consegue prever -- vale como aviso para
outros usuários com virtualização desabilitada na BIOS. Depois disso,
o resto do roteiro (`wsl --install -d Ubuntu`, `pip install
pycolmap-cuda12`, `has_cuda: True`) funcionou exatamente como
documentado, sem precisar instalar CUDA Toolkit manualmente.

1. No Windows (driver NVIDIA normal já instalado, nenhum driver Linux
   dentro do WSL2): `wsl --install -d Ubuntu` (PowerShell como
   administrador), reiniciar se pedido.
2. Dentro do Ubuntu/WSL2: instalar o `cuda-toolkit-13-x` da NVIDIA
   especificamente (não `sudo apt install cuda` genérico -- isso instala
   um driver Linux que quebra o stub de GPU do WSL2).
3. Criar um ambiente Python dentro do WSL2 (`python3 -m venv .venv`),
   `pip install pycolmap-cuda12` (em vez do `pycolmap` normal) mais as
   outras dependências do HTRMapper (`laspy`, `rasterio`, `scipy`).
4. Copiar/clonar o código-fonte do HTRMapper para dentro do WSL2 também
   (ou acessá-lo direto em `/mnt/c/...`, já que o WSL2 enxerga o disco do
   Windows) e instalar com `pip install -e .` (sem o extra `[gui]` --
   dentro do WSL2 só as Fases 4-6 via CLI são necessárias, não a
   interface gráfica).
5. Rodar, por exemplo: `htrmapper dense /mnt/c/Users/.../projeto.json
   --workdir /mnt/c/Users/.../work_dense --quality alta`, apontando
   `--output`/`--workdir` diretamente em `/mnt/c/...` -- esses são
   argumentos digitados pelo usuário, não precisam de tradução (só os
   caminhos que já estavam gravados dentro do `projeto.json` do Windows
   precisam, e isso o HTRMapper já faz sozinho).

## 24. Nuvem densa esparsa no voo real (fazenda 20046) — primeiras hipóteses (ver seção 25 para a causa raiz provável)

A primeira nuvem densa real (seção 15/23: 56 fotos, RTX 3060 Ti, qualidade
"alta") saiu com **densidade baixa** — ~90.246 pontos sobre uma área
fotografada real de ~103.343 m² (ver seção 17 para como essa área real foi
descoberta, comparando com um export do Pix4Dfields do mesmo voo), ou
seja, **~0,87 pontos/m²**. MVS tipicamente entrega dezenas a centenas de
pontos/m² em imagens aéreas de boa sobreposição — essa diferença de
1-2 ordens de grandeza é real, não um artefato de cálculo, e foi o que
produziu o DEM "borrado"/sem relevo reconhecível que o usuário viu.

**Duas causas candidatas, pesquisadas mas não validadas** (este ambiente
de desenvolvimento não tem GPU para testar de verdade):

1. **`num_patch_match_src_images`** (parâmetro do próprio
   `pycolmap.undistort_images`, nunca antes configurado por este
   código): controla quantas imagens-candidatas cada imagem de referência
   usa no patch-match stereo. O valor padrão do pycolmap (`-1`) resolveu,
   empiricamente, em exatamente **20** candidatas nos logs reais da
   execução de 2026-10-06 (`src_image_idxs` com 20 entradas). Mais
   candidatas é estritamente mais dado disponível para o filtro de
   consistência geométrica decidir se um ponto é confiável — nunca menos.

2. **`filter_min_triangulation_angle`** (padrão do COLMAP: 3 graus) e
   **`filter_min_ncc`** (padrão: 0,1), ambos parâmetros de
   `PatchMatchOptions`: a própria documentação FAQ do COLMAP recomenda
   reduzir o ângulo mínimo de triangulação quando as imagens são
   capturadas de longe em relação à cena — exatamente o caso de um voo
   nadir a ~100m de altura, onde ângulos de triangulação genuinamente
   pequenos (mas ainda válidos) são uma característica estrutural da
   imagem aérea top-down, não necessariamente ruído. É plausível que o
   filtro padrão esteja descartando pontos de terreno reais, não só
   blunders.

**O que foi feito agora** (sem mudar nenhum comportamento padrão, já que
nenhuma das duas hipóteses pôde ser validada sem GPU real): os três
parâmetros foram expostos como campos opcionais em `MvsConfig`
(`num_patch_match_src_images`, `filter_min_triangulation_angle`,
`filter_min_ncc`), todos com padrão `None` ("usa o padrão do próprio
pycolmap, comportamento inalterado"), e novas flags de CLI
(`--num-patch-match-src-images`, `--filter-min-triangulation-angle`,
`--filter-min-ncc`) em `htrmapper dense`. Um teste de regressão confirma
que, sem especificar nada, o comportamento é idêntico a antes (passa `-1`
para `num_patch_match_src_images`, não toca nos filtros); outro confirma
que os três valores realmente chegam até as chamadas reais do pycolmap
quando especificados.

**Próximo passo real** (precisa do usuário, com GPU): rodar o mesmo voo
(fazenda 20046) algumas vezes variando esses parâmetros (ex.:
`--num-patch-match-src-images 40`, depois
`--filter-min-triangulation-angle 1.5`) e comparar a densidade de pontos
resultante (`htrmapper dense` já reporta `num_points`) — e então decidir,
com números reais, se vale mudar os valores padrão do projeto. Nunca
assumir que um desses ajustes "resolve" sem essa validação.

## 25. Causa raiz da nuvem densa esparsa: precisão `float32` com coordenadas UTM absolutas (corrigido e confirmado em voo real)

Revisão completa do pipeline em 2026-10-07, depois dos sintomas da seção
17 (ortomosaico borrado, área cortada a oeste e "inventada" a leste).

**O problema**: depois da Fase 2 (`sfm/pipeline.py`, `_georeference`) a
reconstrução vive em coordenadas absolutas do CRS do projeto — no voo
real, X ≈ 260.320 m e Y ≈ 7.548.601 m — e a Fase 4 entregava esse modelo
direto ao MVS do COLMAP, que guarda poses de câmera e pontos fundidos em
precisão simples (`float32`). Em Y ≈ 7,5 milhões, o menor passo
representável em `float32` é **0,5 m** (em X ≈ 260 mil, 1,6 cm). Um erro de
pose de ~0,5 m a ~100 m de altura desloca cada pixel em ~5 px, muito além
dos limiares de 1–2 px do filtro de consistência geométrica do
patch-match e da fusão.

**Os sintomas reais batiam com isso, um por um** (nenhum deles explicado
pelas hipóteses anteriores da seção 24):
- No log da fusão em qualidade "alta", as 27 primeiras imagens somaram
  ~600 pontos; ~90 mil pontos vieram só das imagens de índice 47–55 —
  **só ~9 de 56 imagens geraram mapas de profundidade utilizáveis**.
- "Média" (1600px) gerou **mais** pontos que "alta" (3200px): 158 mil
  contra 90 mil. O normal é o contrário; com um erro métrico fixo, a
  resolução menor transforma o mesmo erro em menos pixels.
- A extensão do DEM/ortomosaico (= caixa dos pontos) ficou deslocada
  ~90 m para leste em relação ao export do Pix4D do mesmo voo: começa ~70 m
  mais a leste (o "bico" cortado a oeste) e vai ~117 m além (a área
  "esticada" à direita, preenchida por extrapolação do DEM).

Ressalva honesta: o uso de `float32` pelo MVS do COLMAP vem do
conhecimento do código-fonte do COLMAP (classe de imagem do MVS e
`PlyPoint` da fusão), não de um teste executado aqui — este ambiente não
tem GPU. A correção abaixo é barata e correta de qualquer forma (é o que o
ODM faz), mas a confirmação de que **era** a causa vem só da próxima
execução real: se quase todas as 56 imagens passarem a contribuir pontos
na fusão, era isso.

**Correção** (`mvs/dense.py`): antes da desdistorção, o modelo é
transladado por uma origem local (média das posições das câmeras,
arredondada ao metro, só X/Y) e gravado em `workdir/sparse_local`; todo o
MVS roda nesse referencial. A origem é somada de volta só nas fronteiras:
- o LAS exportado (`_export_to_las(..., offset=...)`), em `float64`;
- uma cópia georreferenciada do modelo desdistorcido
  (`dense/sparse_georef`), que é o que a Fase 6 consome — a ida e volta é
  exata, porque o modelo esparso é gravado em `float64`.

O `fused.ply` nativo e o workspace denso ficam no referencial local; a
origem fica registrada em `workdir/local_origin.json` e em
`MvsSummary.local_origin_m` (zero para execuções anteriores a isso).

**Validado aqui** (sem GPU): um teste roda a Fase 4 inteira sobre a cena
sintética (que fica em Y ≈ 8,2 milhões, mesma ordem do voo real), com o
patch-match substituído por nada e a fusão devolvendo os próprios pontos
do modelo local. Ele confirma que o que entra no COLMAP está em
coordenadas pequenas (< 1 km) e que o LAS e o modelo da Fase 6 voltam ao
CRS do projeto com erro < 1 mm. Com a correção desligada, o mesmo teste
falha (o COLMAP receberia coordenadas de 8,2 milhões).

**Confirmado em hardware real (2026-10-08)**: mesmo voo (fazenda 20046,
56 fotos), mesma qualidade "media", RTX 3060 Ti, código com a correção:
**2.580.544 pontos** fundidos em 32m31s — contra 158.330 pontos (30m37s)
antes da correção, e 174.897 num teste de controle sem a correção mas com
`--num-patch-match-src-images 40`. Ou seja, ~16x mais pontos pelo mesmo
custo, enquanto o aumento de imagens-candidatas (a hipótese da seção 24)
rendeu só ~10%. Isso confirma a precisão `float32` como causa raiz da
nuvem esparsa.

**Outros gaps encontrados na mesma revisão** (prioridade decrescente;
1 e 2 resolvidos na seção 26):
1. Ortomosaico: a mistura entre fotos é uma média simples — o peso
   `_feather_weight` só cai nos 10% da borda, então no miolo de cada foto
   todas as fotos sobrepostas têm peso 1. Qualquer erro de altura no DEM
   vira borrão. O padrão do Metashape ("Mosaic") escolhe a foto mais
   vertical por região e mistura só nas emendas.
2. Extensão do DEM/ortomosaico vem da caixa dos pontos densos (com
   extrapolação por vizinho mais próximo dentro dela), não da área
   realmente coberta pelas fotos.
3. `RtkStdLon/Lat/Hgt` por imagem é lido no import mas ignorado no ajuste
   GNSS (Fase 3 usa um sigma global); o resíduo GNSS de 71 cm do voo real
   também merece investigação (ex.: atraso entre disparo e registro GNSS,
   se o resíduo acompanhar a direção do voo).
4. Altitude do EXIF usada sem distinção elipsoidal/ortométrica; SIRGAS
   2000 tratado como igual a WGS84 (~0,5 m de diferença hoje pelo
   movimento da placa).

## 26. Ortomosaico: foto mais vertical por pixel, corte da borda oblíqua e formato otimizado

Depois da correção da seção 25, o mesmo voo (fazenda 20046) passou a
cobrir o talhão inteiro, na posição certa em relação ao export do Pix4D.
Na comparação lado a lado ainda havia três diferenças, todas corrigidas
aqui a pedido do usuário:

**1. Mistura por média → foto mais vertical por pixel.** O peso antigo
(`_feather_weight`) só caía nos 10% da borda de cada foto; no miolo todas
as fotos sobrepostas tinham peso 1, então cada pixel era a média simples
de 5–15 vistas. Qualquer erro de altura do DEM (copa da cana, árvores)
desloca o pixel numa direção diferente em cada vista, e a média vira
borrão. Agora cada pixel vem da câmera que o vê mais perto da vertical
(menor ângulo entre o raio da câmera e a vertical), a mesma ideia do modo
"Mosaic", que é o padrão do Metashape: é a vista em que esse deslocamento
é menor. Só onde as duas melhores vistas diferem por menos de
`SEAM_BLEND_DEG` (2°) elas são misturadas, para a emenda não ficar dura —
a ~100 m de altura isso dá uma transição de ~3–4 m. Implementado em uma
passada só por tile, guardando a melhor e a segunda melhor vista por
pixel (memória continua limitada ao tile). `feather_fraction`/
`--feather-fraction` foram removidos.

**2. Corte da borda vista só de lado.** O nosso mosaico ia ~119 m mais a
leste e ~59 m mais ao sul que o do Pix4D, e era exatamente ali que
apareciam as faixas esticadas: chão visto só no cantinho das fotos mais
externas, em ângulo muito inclinado. Agora um pixel cuja melhor vista
ainda é mais oblíqua que `OrthoConfig.max_view_angle_deg` (padrão 30°;
uma foto nadir do Mavic 3M alcança ~28° no lado curto) fica vazio.
CLI: `--max-view-angle`.

**3. Formato do arquivo.** O usuário vinha rodando um script próprio no
QGIS só para reorganizar o GeoTIFF (blocos, compressão, pirâmides); o
arquivo cru de 1,09 GB caía para 355 MB. Agora a Fase 6 já grava assim:
RGB de 3 bandas, blocos 512×512, compressão JPEG (YCbCr, qualidade 90,
escolha do usuário, como o Pix4D faz), pirâmides internas (AVERAGE) e
**máscara interna sem perda** marcando a área vazia. A combinação JPEG +
NoData=0 foi descartada de propósito: o JPEG altera valores perto de
zero, então sombras válidas virariam transparentes (furos) e a borda
vazia viraria uma moldura cinza. O usuário escolheu a máscara entre as
opções. Um experimento isolado antes da implementação confirmou, com o
GDAL 3.10: máscara lida de volta idêntica, erro médio do JPEG de ~1/255
dentro da área válida e pirâmides respeitando a máscara. A visualização
da GUI passou a ler RGB + máscara, o que serve tanto para os arquivos
novos quanto para os RGBA antigos.

**Validado** (cena sintética):
- Uma das câmeras teve a imagem trocada por vermelho puro. No ponto nadir
  dela, o mosaico sai vermelho; no ponto nadir da vizinha (ainda dentro da
  foto vermelha, mas vista mais na vertical pela vizinha), sai a cor
  verdadeira do terreno, sem tom vermelho. Forçando a mistura em toda a
  sobreposição, o mesmo teste falha (o pixel sai rosado).
- Com 10°, todo pixel mantido fica a no máximo tan(10°) × altura de voo
  do ponto nadir de alguma câmera. A contagem de pixels válidos cresce de
  10° para 30° e de 30° para sem corte.
- O arquivo sai com 3 bandas, JPEG/YCbCr, blocos 512×512, máscara por
  dataset e pirâmides, e a máscara bate com `num_valid_pixels`.
- Os testes anteriores (cores conhecidas com erro médio < 8/255, tile
  gigante igual a tile de 8 px byte a byte) continuam passando com JPEG.

**Validado em voo real** (fazenda 20046, `ortho_v2`): visualmente aprovado
pelo usuário, 53 MB (o JPEG só gasta bytes nos 32% de pixels válidos;
blocos mascarados quase não ocupam espaço), transparência correta, 45 das
56 câmeras usadas. Mas o mosaico ficou ~70 cm deslocado em relação ao do
Pix4D — causa e correção na seção 27.

## 27. Precisão absoluta: deslocamento de ~70 cm em relação ao Pix4D

O mosaico do voo real ficou ~70 cm deslocado do Pix4D no mesmo ponto. O
uso final (restituição de linha de cana para piloto automático de
colhedora) exige precisão absoluta, então a cadeia inteira foi revisada,
com diagnósticos rodados pelo usuário sobre a reconstrução real (scripts
só de leitura, sem alterar o projeto).

**Diagnóstico.** As câmeras com RTK FIX (desvio informado pelo receptor
de ~5 mm) ficaram, depois do ajuste, a 30 cm (X) / 26 cm (Y) / 17 cm (Z)
RMSE da própria posição GNSS, com máximo de 115 cm. Não era um
deslocamento rígido (a média é zero por construção: a translação global
não altera a reprojeção, então o ótimo centra o bloco no GNSS). O resíduo
variava suavemente ao longo das faixas: o bloco estava entortado.

Variantes do ajuste sobre a mesma reconstrução da Fase 2 (RMSE das fotos
FIX, X / Y / Z):

| Variante | RMSE | Máx. XY |
|---|---|---|
| A. como era (intrínsecos fixos, sigma único) | 30,0 / 26,3 / 17,5 cm | 114,8 cm |
| B. + sigma por foto (desvio RTK) | 31,6 / 18,3 / 7,4 cm | 72,2 cm |
| C. + distorção livre | 18,9 / 11,7 / 5,2 cm | 44,1 cm |
| E. + ponto principal livre | 2,9 / 2,0 / 1,2 cm | 7,5 cm |
| H. + focal livre (padrão novo) | 2,3 / 2,2 / 1,0 cm | 6,4 cm |
| F. modelo FULL_OPENCV (k3..k6) | 2,6 / 2,3 / 1,1 cm | 6,4 cm |

**Causa principal: ponto principal fixo no centro da imagem.** A Fase 2
(COLMAP) não refina o ponto principal, e a Fase 3 mantinha a lente
inteira fixa. O ajuste o colocou em (2667,4; 1959,6), a 27 / 18 px do
centro. A 100 m de altura, 27 px de erro inclinam todo raio em
27/3721 rad, movendo o chão ~73 cm, o que bate com o deslocamento visto.
Faixas em sentidos opostos (rumo 245° e 65° neste voo) veem esse
deslocamento em direções opostas, então o bloco entorta para conciliar e
o mosaico sai no lugar errado. Não dá para separar, numa única altura de
voo, ponto principal de um braço de alavanca horizontal antena-câmera
(os dois movem o chão relativo ao GNSS de forma que gira com o rumo), mas
para o mosaico só importa o efeito combinado, e é esse que o ajuste estima.

Confirmações cruzadas:
- Com a focal livre, ela convergiu para 3723,9 px. O `CalibratedFocalLength`
  de fábrica gravado pela DJI no XMP dessa câmera é 3725,15 px (0,03% de
  diferença).
- Decomposto na direção de voo, o resíduo da variante H tem média de
  +0,2 cm ao longo e +0,9 cm lateral (desvio ~2 cm). Isso descarta atraso
  entre o disparo e o RTK, que apareceria como erro sistemático ao longo do
  voo.
- O modelo FULL_OPENCV (com k3, mas também os termos racionais k4..k6) não
  ganhou nada sobre o OPENCV e deixou os coeficientes sem sentido físico
  (k1 = −5,5, k2 = 11), sinal de superparametrização. Ficou de fora.

**Correções implementadas na Fase 3** (`ba.weighted_bundle_adjustment`):
1. Focal, ponto principal e distorção são refinados por padrão, com o GNSS
   como referência. A degenerescência da seção 14 é real, mas aparece no
   voo sintético de 6 fotos (focal indo de ~222 para ~199 px, ponto
   principal 10% da imagem fora do centro). Por isso cada parâmetro tem
   trava: se a focal mudar mais de 5% ou o ponto principal sair mais de 2%
   do tamanho da imagem do centro (fisicamente implausível numa câmera de
   drone real; o voo real ficou em 0,07% e 0,5%), o ajuste é refeito com
   aquele parâmetro fixo e o motivo aparece em `intrinsics_notes`. CLI:
   `--fix-intrinsics` volta ao comportamento antigo.
2. Sigma por foto: a `CameraAccuracy` configurada vira um piso e, quando o
   receptor informa um desvio maior (`RtkStdLon/Lat/Hgt`), vale o maior.
   As duas primeiras fotos do voo real eram RTK FLOAT (desvio de 17–47 cm,
   GNSS errado em 3–4 m) e, pesadas como FIX, puxavam o começo do bloco.
3. O RMSE reportado considera só fotos RTK FIX (ou todas, se o voo não tem
   flag RTK, como em PPK). O resíduo de uma foto FLOAT mede o GNSS ruim
   dela, não o ajuste. As fotos excluídas são listadas.
4. **Falha silenciosa do COLMAP.** Lendo o código-fonte do
   `PosePriorBundleAdjuster` (COLMAP 4): antes de ajustar, ele realinha a
   reconstrução aos priors com um RANSAC de Sim3 e, **se esse
   alinhamento falha, descarta todos os priors GNSS** e roda um ajuste só
   de reprojeção, avisando apenas no log. O limiar automático (~2,8 σ, uns
   6 cm com σ = 2 cm) fazia isso acontecer de vez em quando no voo
   sintético com uma foto ruim, deixando as câmeras a 21 m do lugar. Era
   essa a causa dos testes "instáveis" do ajuste registrados antes: em 5
   rodadas seguidas da suíte do ajuste, 0 falhas depois da correção. Agora
   o limiar é 5 m (a entrada já vem alinhada ao mesmo GNSS pela Fase 2, com
   desvios de até ~1,2 m no voo real; o passo só precisa rejeitar erros
   grosseiros), e o número de resíduos do Ceres é conferido (3 por prior,
   2 por observação). Se o COLMAP tiver descartado os priors, o ajuste
   falha com erro claro em vez de seguir com um resultado inválido.
5. Tolerâncias do solver mais apertadas (função 1e-10, até 500 iterações).
   As variantes acima foram validadas assim; com a tolerância antiga
   (1e-6) a variante C parou no limite de 200 iterações.

**Correção na Fase 6: meio pixel.** O COLMAP põe a origem da imagem no
canto do pixel superior esquerdo, então o centro desse pixel é (0,5; 0,5).
Por isso o ponto principal padrão é largura/2 × altura/2. O ortomosaico
amostrava essas coordenadas como índice de array, onde o centro do pixel
é inteiro, e lia cada foto meio pixel deslocada (~1 cm no GSD nativo,
numa direção que gira com o rumo de cada câmera). Teste com câmera do
próprio pycolmap: o raio do eixo óptico tem que ler a média dos quatro
pixels centrais. Ele falha sem a correção e passa com ela.

**Sistema de coordenadas, conferido:**
- EXIF GPS da DJI: minutos com 7 casas decimais (~0,2 mm), mesmo valor
  do `GpsLatitude/GpsLongitude` do XMP. A leitura não perde precisão.
- WGS84 → SIRGAS 2000 / UTM 23S pelo PROJ: transformação nula ("Inverse
  of SIRGAS 2000 to WGS 84 (1)"). No ponto do voo, EPSG:31983 e EPSG:32723
  (o CRS do Pix4D) diferem em 0,07 mm, então o CRS não explica a diferença
  com o Pix4D. A altura passa inalterada.
- Rodar o ajuste em coordenadas locais (só XY transladado, como a Fase 4)
  deu resultado idêntico ao UTM absoluto. O próprio COLMAP normaliza a
  reconstrução antes de ajustar. Já transladar também o Z para câmeras em
  z ≈ 0 quebrou o ajuste sintético, mais um motivo para o referencial local
  da Fase 4 manter Z = 0 de origem.
- Pendente: a altitude da DJI (`AltitudeType = RtkAlt`) vem do referencial
  das correções RTK, e não está confirmado se é elipsoidal ou ortométrica.
  Isso não afeta XY do mosaico nem volumes, só o valor absoluto das cotas
  do DEM. Basta comparar uma cota do DEM com a do Pix4D no mesmo ponto.

**Não feito ainda (candidatos, por ordem de impacto esperado):**
- DEM: hoje é interpolação linear da nuvem crua, sem filtro de ruído. O
  ruído do DEM desloca pixels do mosaico perto da borda das fotos (erro de
  cota × tangente do ângulo de visada).
- Fase 2 extrai features em imagens reduzidas a 2000 px (de 5280). O
  resíduo já caiu ao nível do ruído do RTK, então só vale testar em
  resolução cheia se a verificação contra pontos de controle pedir.
- Usar a calibração de fábrica da DJI (`DewarpData`: fx, cx/cy, k1, k2,
  p1, p2, k3) como ponto de partida da Fase 2.
- Prova de precisão absoluta: pontos de verificação medidos em campo com
  rover RTK. Concordar com o Pix4D não prova precisão, porque ele usa o
  mesmo GNSS.

**Confirmado em voo real** (fazenda 20046, `ortho_v3`): o ajuste rodou com
focal, ponto principal e distorção livres, sem nenhuma trava acionada;
RMSE das câmeras RTK FIX de 2,3 / 2,2 / 1,0 cm (máx. 6,4 cm), com as duas
fotos FLOAT fora do RMSE. Comparado no QGIS, o mosaico ficou praticamente
idêntico ao do Pix4D; o deslocamento de ~70 cm sumiu.

## 28. Revisão contra Metashape e Pix4D; DEM refeito; bug de CRS

Com a precisão confirmada (seção 27), revisamos o pipeline contra o que
Agisoft e Pix4D documentam, para achar diferenças que valem
implementação. Fontes primárias: o manual oficial do Metashape
Professional 2.2 (PDF da Agisoft, lido por inteiro nas partes de
alinhamento, otimização, modelo de câmera, nuvem densa, DEM e
ortomosaico) e as páginas de "processing options" do Pix4Dmapper.

### O que eles fazem e onde estamos

| Tema | Metashape | Pix4Dmapper | HTRMapper |
|---|---|---|---|
| Features no alinhamento | High = imagem original; Highest = 2x | Keypoints "Full" = escala original | imagem reduzida a 2000 px (~2,6x menor por lado) |
| Pré-seleção de pares | por posição GNSS; para lavoura, recomenda **só** por referência | Aerial Grid / Corridor | espacial por GNSS (equivalente) |
| Padrões repetidos (linhas de cana) | "Guided image matching" para vegetação | "Geometrically verified matching" | verificação geométrica padrão do COLMAP |
| Modelo de lente | f, cx, cy, K1–K3, P1, P2 (+K4, B1, B2 opcionais) | R1–R3, T1, T2 | fx, fy, cx, cy, k1, k2, p1, p2 (sem k3) |
| Lente em voo plano | "Adaptive camera model fitting" evita divergência | "All Prior" segura perto do inicial; método "Alternative" para lavoura nadir com GNSS preciso | refina tudo com travas rígidas (seção 27) |
| Peso do GNSS | precisão por câmera, por eixo | por imagem | por imagem, com o desvio RTK (seção 27) |
| Limpeza de tie points | gradual selection (USGS: incerteza ~10–15, projection accuracy ~2–3, reprojeção 0,3 px) + reotimização | — | só o filtro do COLMAP (4 px) |
| Nuvem densa | Ultra = original; High = 1/2; Medium = 1/4 | padrão 1/2, multiescala | "media" = 1600 px (~1/3,3) |
| Ruído da nuvem | filtro por componente conectado no depth map; confiança por ponto | — | consistência geométrica + mínimo de vistas da fusão do COLMAP |
| Resolução do DEM | a da própria nuvem | 1x GSD | era 2,5x o espaçamento (0,63 m no voo real) |
| Ruído no DEM | — | "noise filtering": altitude trocada pela mediana dos vizinhos | não havia |
| Lacunas do DEM | padrão: só onde alguma foto vê; "Extrapolated" é opcional | triangulação recomendada para lavoura | extrapolava até o retângulo envolvente |
| Mistura no ortomosaico | "Mosaic": multibanda junto da costura | — | foto mais vertical + transição de 2° (seção 26) |
| Convenção de pixel | centro do primeiro pixel em (0,5; 0,5) | — | idem desde a seção 27 (o manual confirma a correção) |
| Altitude DJI | depende do datum escolhido | supõe EGM96 salvo se a câmera estiver no banco como elipsoidal | usada como está (ver abaixo) |

### Implementado agora: DEM

Banco de testes reprodutível com superfície conhecida: talhão de
300 × 250 m com 16 pontos/m² (a densidade do voo real), linhas de cana
de 1,5 m com 15 cm de relevo, uma valeta de 2 m × 80 cm, um buraco de
1.000 m² sem pontos, uma ponta do talhão sem dados, ruído de 3 cm e 0,5%
de pontos espúrios de 0,5 a 3 m.

| | Antes | Depois |
|---|---|---|
| Erro RMS (resolução automática) | 10,5 cm | 5,0 cm |
| p99 do erro | 41 cm | 14 cm |
| Células com erro > 30 cm | 1,25% | 0,08% |
| Maior pico | 2,84 m | 0,89 m (célula metade dentro, metade fora da valeta) |
| Buraco de 1.000 m² | 14,1 cm | 12,6 cm (as linhas de cana sozinhas somam 10,6) |
| Terreno inventado fora dos dados | 4.621 m² | 498 m² (só a margem de 2 m) |
| Tempo | 29 s | 1,2 s |

Como ficou (`dem.generation`):
1. mediana dos pontos de cada célula, em vez de interpolação linear dos
   pontos crus (um ponto errado num triângulo contaminava toda a vizinhança);
2. filtro de picos à la Pix4D: célula que se afasta da mediana 5 × 5 por
   mais de 4 sigmas robustos (mínimo 15 cm) é descartada e reinterpolada,
   **desde que** forme um grupo pequeno (≤ 1 m²). Sem essa condição, um
   aterro de 1 m de largura foi apagado inteiro; com ela, sobrevive intacto
   e 99% dos picos injetados ainda saem;
3. lacunas pequenas pela média dos vizinhos; buracos grandes linearmente a
   partir da borda real (crescer o buraco por anéis enviesava pela
   inclinação: 23,7 cm contra 13,4 cm);
4. NoData fora da área reconstruída (margem de 2 m ou 2 células, mais
   buracos cercados por dados), como o modo padrão do Metashape. O
   ortomosaico estende sozinho a cota mais próxima onde a câmera ainda vê o
   chão, então não perde área;
5. resolução automática de 2,0x o espaçamento (~4 pontos por célula,
   ~2% de células vazias), grade alinhada a múltiplos da resolução,
   GeoTIFF em blocos 256 com DEFLATE.

### Bug corrigido: CRS de exportação

A nuvem LAS e o DEM eram gravados com o rótulo do CRS de exportação
(`export_epsg`), mas as coordenadas continuavam no CRS do projeto, porque
nada reprojeta. Com exportação em EPSG:4674 (geográfico, graus), um DEM em
UTM (metros) sairia rotulado como graus, no lugar errado do mundo. Um
teste antigo exigia exatamente isso. Agora tudo declara o CRS em que as
coordenadas realmente estão, e a CLI avisa que a reprojeção para o CRS de
exportação ainda não existe. No voo real nada mudou, porque `export_epsg`
não estava configurado.

### Altitude da DJI

O XMP do voo real traz `Version = 1.6` e `AltitudeType = RtkAlt`. Relatos
de usuários (não há documento oficial da DJI) dizem que a partir da versão
1.5 do XMP a altitude é elipsoidal, e é o que as correções RTK entregam.
O Pix4D já tratou M3E como EGM96 por engano até a 4.8.3. Então as cotas do
nosso DEM são, muito provavelmente, **elipsoidais** (SIRGAS 2000). Para
cota ortométrica é preciso aplicar o modelo geoidal do IBGE (hgeoHNOR2020).
Isso não afeta XY do mosaico nem volumes.

### Candidatos que ficaram para decisão do usuário

1. Features em resolução cheia no alinhamento (padrão de Metashape e Pix4D).
2. Limpeza de tie points em etapas + reotimização (workflow USGS/Metashape).
3. k3 no modelo de lente (o COLMAP só oferece k3 junto com os termos
   racionais k4–k6, que ficaram superparametrizados na seção 27).
4. Mistura multibanda nas costuras do ortomosaico (visual).
5. Conversão opcional para altitude ortométrica (hgeoHNOR2020).
6. Pontos de controle/verificação (marcadores com coordenada de campo), a
   prova de precisão absoluta que nem Pix4D nem nós temos hoje no projeto.
7. Reprojeção real para o CRS de exportação.

## 29. Melhorias escolhidas da revisão: tie points, resolução cheia, cota ortométrica, pontos de verificação

O usuário escolheu quatro candidatos da seção 28.

### Limpeza de tie points em etapas (Fase 3)

Equivalente automático do ciclo "gradual selection" + "Optimize Cameras"
do Metashape, como o script do USGS faz. Depois do ajuste (com as travas
de lente), até 3 passes: saem as observações com reprojeção acima de
max(3 × mediana do erro dos pontos, 1 px), os pontos triangulados com
ângulo menor que 1,5° (o análogo da "reconstruction uncertainty") e os
vistos em menos de 3 fotos; o ajuste roda de novo. O piso de 1 px faz o
filtro tirar erros grosseiros, não perseguir um número bonito (o voo real
tem média de ~0,5 px). Se a limpeza fizer a lente sair dos limites, o
resultado sem limpeza é mantido e o motivo é anotado. CLI:
`--no-tie-point-cleanup` desliga.

Validado com 40 observações corrompidas em 15 px no voo sintético: sem
limpeza, 39 continuam acima de 5 px depois do ajuste; com limpeza, nenhuma,
e o erro médio cai de 0,31 para 0,08 px.

**Bug encontrado no caminho:** o erro de reprojeção "final" que o `adjust`
reportava era o da Fase 2. O COLMAP guarda o erro de cada ponto num cache
que o bundle adjuster não atualiza, e `compute_mean_reprojection_error()`
lê esse cache. Por isso todas as variantes do voo real mostravam
exatamente 0,499 px. Agora o cache é atualizado antes da leitura (também
na Fase 2), e um teste compara o valor reportado com um recálculo
independente, projetando cada observação.

### Alinhamento em resolução cheia

Já existia como `htrmapper align --max-image-size -1` (seção 13). Medido
no voo real da fazenda 20046 (56 fotos), ambos com a limpeza de tie points:

| | 2000 px (padrão) | resolução cheia |
|---|---|---|
| Tie points após a limpeza | 62.306 | 59.361 |
| Reprojeção final | 0,330 px | 0,326 px |
| RMSE GNSS XY / Z / total | 2,88 / 0,98 / 3,04 cm | 2,79 / 1,02 / 2,97 cm |
| Erro máximo | 6,15 cm | 6,06 cm |
| Focal / cx / cy final (px) | 3724,73 / 2667,79 / 1959,61 | 3724,56 / 2667,71 / 1959,48 |

A lente muda menos de 0,2 px (abaixo de 0,5 cm no chão) e o RMSE 0,7 mm:
os dois já estão no piso do RTK (~2 cm), e o gargalo não é a precisão dos
tie points. O padrão continua 2000 px, mais rápido.

Na mesma comparação, o DSM do Pix4D do voo (EPSG:32723, elipsoidal) ficou
+0,29 m acima do `dem_v4` (mediana; desvio robusto 0,22 m). É compatível
com a correlação focal × altura de voo só nadir sem pontos de controle,
que não desloca o ortomosaico em XY: os dois ortomosaicos batem.

### Cota ortométrica (`geo.vertical`, `htrmapper orthometric`)

O PROJ não distribui grade geoidal brasileira. A grade oficial do IBGE
(`hgeoHNOR2020__IMBITUBA__fator-conversao.txt`, passo de 5', colunas
longitude 0–360°, latitude e FC em metros) é baixada uma vez do FTP público
do IBGE para `~/.cache/htrmapper`, ou lida de uma cópia local
(`--geoid-grid`). H = h − FC; na fazenda 20046, FC ≈ −6,03 m, então a cota
normal fica ~6 m acima da elipsoidal.

O resultado é um **arquivo à parte** (`DemSummary.orthometric_raster_path`).
O DEM do projeto continua elipsoidal de propósito: as câmeras estão em
altitude elipsoidal, e um ortomosaico calculado sobre cotas normais erraria
o chão em ~6 m, deslocando os pixels da borda em até ~3,5 m (6 m × tan 30°).
Células fora do modelo são erro, nunca deixadas elipsoidais em silêncio; um
DEM já convertido (tag `HTRMAPPER_VERTICAL_REFERENCE`) não é convertido de
novo. Os testes usam um recorte real da grade do IBGE em volta do talhão.

### Pontos de verificação (`geo.checkpoints`, `htrmapper checkpoints`)

Concordar com o Pix4D prova consistência, não precisão, porque os dois partem
dos mesmos geotags. A prova são alvos medidos em campo com rover RTK.

- Entrada: CSV do rover (`nome;E;N;h`, no CRS do projeto, com `;`/`,`/tab
  e vírgula decimal), opcionalmente com `E_ortho;N_ortho` medidos à mão no
  QGIS.
- Horizontal: num raio de 1 m da coordenada de campo, o alvo é a mancha
  clara (ou escura, `--target-color dark`) de 0,05–4 m² mais próxima,
  separada do fundo por um limiar no meio do caminho entre a mediana da
  janela e o percentil 1%/99%. O centro da mancha é a posição medida.
  O limiar de Otsu foi testado e descartado, porque divide ao meio a
  textura das linhas de cana quando o alvo ocupa pouco da janela. Contraste
  menor que 30 níveis contra um anel de 20 cm em volta do alvo vira "não
  encontrado", nunca um palpite.
- Vertical: cota do DEM (bilinear) menos a altura de campo, usando o DEM
  ortométrico quando `--field-heights orthometric`.
- Saída: tabela por ponto, RMSE, média (viés) e máximo, mais um CSV com
  `;` para planilha.

Validado com ortomosaico sintético texturizado e alvos deslocados de valores
conhecidos (12 cm L, 7 cm S): o erro medido bate dentro de 1 GSD; alvo
escuro, medida manual, ausência de alvo e um objeto claro concorrente
dentro do raio também são cobertos.

Ainda não implementado: usar os alvos como **pontos de controle** dentro do
ajuste (marcadores observados nas fotos, como no Metashape). O caminho
técnico existe no pycolmap (um ponto 3D fixo com as observações do alvo
em cada foto), mas exige localizar o alvo em cada imagem.

