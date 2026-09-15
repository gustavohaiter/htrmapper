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

## 11. Fase 1 — o que está implementado neste commit

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
