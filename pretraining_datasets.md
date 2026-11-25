# VHR pretraining image sets (continual DINOv3 adaptation)

Open image sets used to continue training a DINOv3 encoder on very high resolution (VHR)
overhead RGB imagery, roughly 1.5 m per pixel or finer. Training is self-supervised, so
only the pictures are used, not the labels. The built dataset is published as WebDataset
shards at [kshitijrajsharma/dinov3-vhr-pretrain](https://huggingface.co/datasets/kshitijrajsharma/dinov3-vhr-pretrain),
with an [interactive coverage map](https://huggingface.co/spaces/kshitijrajsharma/dinov3-vhr-coverage)
that toggles by platform, sensor, and source.

## Contents

1,018,163 non-overlapping 256 x 256 RGB tiles across seven sources. Each tile is a lossless
PNG with a JSON sidecar holding source, license, ground sample distance, coordinates, and
sensor. Shards hold 10,000 tiles each.

| Source | Tiles | Platform | Sensor | Resolution |
| --- | --- | --- | --- | --- |
| fMoW | 500,008 | Satellite | QuickBird-2, GeoEye-1, WorldView-2/3 | 0.3 to 0.7 m |
| Maxar Open Data | 120,000 | Satellite | WorldView | 0.3 m |
| OpenAerialMap | 100,000 | Drone and aircraft | Mixed contributor | 0.05 to 1 m |
| NAIP (SatlasPretrain) | 100,000 | Aerial ortho | NAIP | ~1 m |
| RAMP | 95,333 | Satellite | WorldView, GeoEye | 0.3 to 0.5 m |
| WorldStrat | 62,822 | Satellite | SPOT 6/7 | 1.5 m |
| Planet (disaster) | 40,000 | Satellite | SkySat, Pelican | 0.5 m |

License for combined use is CC-BY-NC-SA-4.0, the most restrictive of the source terms, so
the dataset is non-commercial and share-alike.

## Sampling

An equal-weight mix draws each source with the same probability at training time, so every
sensor is seen in roughly a one-seventh share regardless of its raw tile count. This upsamples
the small drone and aerial sources and downsamples fMoW, which would otherwise dominate. fMoW
is capped at about 500k of the millions of time-stamped views available; NAIP is capped at
about 0.2 percent of the ~47M SatlasPretrain tiles, since at full scale it would make the
model US-aerial.

## Evaluation holdout

Evaluation uses the pangaea2.0 VHR suite (branch `run`, `scripts/suite.yaml`): spacenet2,
spacenet3, flair2, hot_buildings, xview2, pureforest, mpv4ger, opencanopy, everwatch,
nzcattle_det. None are in the training list, so no labels leak. Three imagery holdouts remove
the only pixel overlaps, applied at build time:

- xView2 / xBD Maxar disaster events (pre-2020) are excluded from the Maxar source, which
  otherwise draws from the same Maxar Open Data bucket.
- RAMP's two SpaceNet-2 subsets (Shanghai, Paris) are excluded, since spacenet2 and spacenet3
  evaluate that imagery.
- The OpenAerialMap mosaic `60d35697eade0f00079b58c6` (Tajikistan) that hot_buildings is built
  from is excluded.

The rest of the suite (French IGN, German aerial, drone) shares no source with the dataset.

## Sensor and platform mix

| Platform | Sensors / sources | Resolution | Sources |
| --- | --- | --- | --- |
| Commercial satellite (Maxar family) | QuickBird-2, GeoEye-1, WorldView-2/3 | 0.3 to 0.7 m | fMoW, Maxar Open Data, RAMP |
| US aerial ortho | NAIP | ~1 m | NAIP (SatlasPretrain) |
| Drone and aircraft | Contributor uploads | 0.05 to 0.5 m | OpenAerialMap |
| Airbus (SPOT) | SPOT 6/7 | 1.5 m (pansharpened) | WorldStrat |
| Planet small-sat | SkySat, Pelican | 0.5 m | Planet (disaster) |

The dataset leans satellite (fMoW, Maxar, RAMP, WorldStrat, Planet) with drone and aerial ortho
as a balanced minority (OpenAerialMap, NAIP). Per-tile sensor for OpenAerialMap is contributor
dependent and often unlabeled. Not covered: Airbus Pleiades at 0.5 m (no open imagery at scale),
European national ortho, and Chinese Gaofen.

## Dropped and excluded

Considered but not in the built dataset:

- **LoveDA**: Google Earth aggregate at 0.3 m, CC-BY-NC-SA-4.0. Not pulled for this build.
- **DeepGlobe**: DigitalGlobe Vivid+ basemap at 0.5 m, research-only terms and Kaggle access.
- **SkyScript**: wanted only for its Planet SkySat share, but it ships as bundled zips with no
  per-source download, so SkySat is pulled directly from the Planet disaster releases instead.
- **Million-AID**: CC-BY-NC-ND (NoDerivatives), a licensing watch-item.
- **xView / xBD, DOTA / DIOR, GID (Gaofen), iSAID**: registration-gated, evaluation-suite
  imagery, academic-only terms, or coarser than VHR.
- **SkySense pretraining data**: proprietary, not released.

## Is it enough?

The nearest works that continue DINOv2 rather than train from scratch are
[ExPLoRA](https://arxiv.org/abs/2406.10973), which continues DINOv2's objective on fMoW-RGB
sub-meter satellite imagery, and [SoftCon](https://arxiv.org/abs/2405.20462), which continues
it on 10 m Sentinel-2. Neither targets sub-meter aerial at this scale. For rough sizing,
continual runs of existing models have used 220k to 780k images (Copernicus-FM, GFM, SoftCon)
with useful gains from around 100k. This dataset lands at 1.02M tiles.

## Sources

- Image sets: [OpenAerialMap](https://openaerialmap.org/) and [AWS registry](https://registry.opendata.aws/openaerialmap/), [fMoW](https://arxiv.org/abs/1711.07846) and [license](https://raw.githubusercontent.com/fMoW/dataset/master/LICENSE), [Maxar Open Data](https://www.maxar.com/open-data), [SatlasPretrain](https://arxiv.org/abs/2211.15660), [RAMP](https://rampml.global/training-data/) and [Source Cooperative](https://source.coop/ramp/ramp), [WorldStrat](https://arxiv.org/abs/2207.06418) and [Zenodo](https://zenodo.org/record/6810792), [Planet disaster data](https://www.planet.com/disasterdata/)
- Models: [SatMAE](https://arxiv.org/abs/2207.08051), [Scale-MAE](https://arxiv.org/abs/2212.14532), [SatDINO](https://arxiv.org/abs/2508.21402), [Satlas](https://arxiv.org/abs/2211.15660), [Clay](https://github.com/Clay-foundation/model), [GFM](https://arxiv.org/abs/2302.04476)
- Scale and eval: [ExPLoRA](https://arxiv.org/abs/2406.10973), [SoftCon](https://arxiv.org/abs/2405.20462), [Copernicus-FM](https://arxiv.org/abs/2503.11849), [DINOv3](https://arxiv.org/abs/2508.10104)
