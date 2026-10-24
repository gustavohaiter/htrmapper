"""The three processing levels of the automatic run (Baixo / Médio / Alto).

Médio is exactly the configuration validated on the real fazenda 20046
flight against Pix4D (ARCHITECTURE.md section 30): default alignment, GNSS
adjustment with tie point cleanup, "media" dense cloud, orthomosaic at the
camera GSD. Baixo trades quality for speed everywhere; Alto only raises the
dense cloud resolution -- the orthomosaic position barely changes (the
cameras already sit at the RTK noise floor), the DEM, point cloud and the
block edges get more detail.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from htrmapper.ba.weighted_bundle_adjustment import BaConfig
from htrmapper.dem.generation import DemConfig
from htrmapper.mvs.dense import MvsConfig
from htrmapper.ortho.orthomosaic import OrthoConfig
from htrmapper.sfm.pipeline import SfmConfig


@dataclass(frozen=True)
class ProcessingLevel:
    key: str
    label: str
    summary: str
    description: str
    sfm: SfmConfig = field(default_factory=SfmConfig)
    ba: BaConfig = field(default_factory=BaConfig)
    mvs: MvsConfig = field(default_factory=MvsConfig)
    dem: DemConfig = field(default_factory=DemConfig)
    ortho: OrthoConfig = field(default_factory=OrthoConfig)

    def parameters_text(self) -> list[str]:
        """Human-readable parameter list, for the report and the log."""
        return [
            f"Alinhamento: até {self.sfm.key_point_limit} pontos-chave por foto, "
            f"fotos reduzidas a {self.sfm.max_image_size} px, {self.sfm.spatial_max_neighbors} vizinhas por foto",
            (
                "Ajuste GNSS: lente refinada (focal, centro óptico, distorção), "
                + (
                    "limpeza de tie points em até 3 passes"
                    if self.ba.tie_point_cleanup
                    else "sem limpeza de tie points"
                )
            ),
            f"Nuvem densa: qualidade '{self.mvs.quality}'",
            "DEM: resolução automática (2x o espaçamento dos pontos), filtro de picos, sem extrapolação",
            (
                "Ortomosaico: pixel = "
                + ("GSD da câmera" if self.ortho.resolution_scale == 1.0 else f"{self.ortho.resolution_scale:g}x o GSD")
                + f", fotos até {self.ortho.max_view_angle_deg:g}° da vertical, JPEG q90 + máscara + pirâmides"
            ),
        ]


LEVELS: dict[str, ProcessingLevel] = {
    "baixo": ProcessingLevel(
        key="baixo",
        label="Baixo",
        summary="Visualização rápida",
        description=(
            "Só para visualização simples, sem esperar muita precisão nem qualidade. "
            "Alinhamento rápido, nuvem densa leve e ortomosaico com pixel de 2x o GSD."
        ),
        sfm=SfmConfig(key_point_limit=8000),
        mvs=MvsConfig(quality="baixa"),
        ortho=OrthoConfig(resolution_scale=2.0),
    ),
    "medio": ProcessingLevel(
        key="medio",
        label="Médio",
        summary="Mapeamento",
        description=(
            "Para mapear: ver e desenhar o talhão, restituir linha, com boa qualidade. "
            "É a configuração validada contra o Pix4D. Ortomosaico com pixel = GSD."
        ),
        mvs=MvsConfig(quality="media"),
    ),
    "alto": ProcessingLevel(
        key="alto",
        label="Alto",
        summary="O mais completo",
        description=(
            "Top de linha: tudo do Médio, com nuvem densa em resolução 2x maior (4x mais pontos). "
            "DEM e nuvem mais detalhados e bordas melhores; bem mais demorado."
        ),
        mvs=MvsConfig(quality="alta"),
    ),
}

DEFAULT_LEVEL = "medio"
