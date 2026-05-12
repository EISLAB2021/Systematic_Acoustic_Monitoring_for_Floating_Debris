# Acoustic-Based Underwater Floating Debris Tracking

Tracking module for an acoustic monitoring system for underwater floating debris. This repository focuses on using spatiotemporal information in acoustic data to estimate debris motion vectors, support behavior analysis, and generate motion-based pseudo-labels.

The debris detection component was developed and is maintained separately by another team member. Please refer to https://github.com/EISLAB2021/Marine_Debris_Detection_YOLOv8.

## Project Background

This work is part of a three-year environmental protection project aimed at developing a systematic acoustic monitoring method for underwater floating debris. The system is designed for low-visibility conditions, such as dark or turbid underwater environments, which are common in most marine environments. Forward-looking sonar (FLS) is used to observe debris targets, and this repository provides the tracking component that estimates target motion from sequential sonar frames.

## Scope

This repository covers:

- motion-vector estimation from adjacent sonar frames
- tracking-oriented visualization of underwater floating debris movement
- behavior analysis based on temporal motion cues
- pseudo-label generation from motion regions
- auxiliary frequency-domain inspection of sonar image quality and texture

## Sonar Hardware

The laboratory system uses an **ARIS Explorer 3000 forward-looking sonar**. Its acoustic imaging output provides the frame sequence used by the tracking pipeline in this repository.

| Property | ARIS Explorer 3000 |
| --- | --- |
| Dimensions | 26 x 16 x 14 cm |
| Weight in air | 5.12 kg |
| Weight in water | 1.55 kg |
| Number of transducer beams | 128 beams |
| Beam width | 0.25 deg |
| Field of view | 30 deg x 15 deg |
| Frame rate | Up to 15 frames/s |
| Range resolution | Down to 3 mm |
| Power consumption | 18 W typical |
| Cable length | Up to 150 m |
| Provided power supply input | 100-240 Vac |
| Provided power supply output | 48 Vdc |
| Maximum power | 60 W |

Field experiments used a **DIDSON forward-looking sonar**, the previous generation of ARIS. The DIDSON configuration used in the field study is summarized below.

| Property | DIDSON |
| --- | --- |
| Operating frequency | 1.8 MHz |
| Beamwidth, two-way | 0.3 deg horizontal by 14 deg vertical |
| Number of beams | 96 |
| Start range | 0.42 m to 12.92 m in 0.42 m intervals |
| Window length | 1.25 m, 2.5 m, 5 m, 10 m |
| Range bin size | 2.5 mm, 5 mm, 10 mm, 20 mm |
| Pulse length | 4.5 us, 9 us, 18 us, 36 us |
| Max frame rate | 4-21 frames/s, depending on window length |
| Field of view | 29 deg |
| Remote focus | 1 m to maximum range |
| Power consumption | 30 W typical |
| Control | Ethernet |
| Display up-link | Ethernet or NTSC video |
| Maximum cable length, 100/10BaseT | 152 m |
| Maximum cable length, Patton extender | 1220 m with local power |

ARIS and DIDSON data are stored in sonar-specific beam/range formats before being remapped into image-like frames for visualization and downstream processing. In particular, ARIS recordings are typically stored in proprietary .aris formats, while DIDSON data are commonly stored in .ddf (DIDSON Data File) formats. Both formats preserve the original acoustic beam and range information acquired by the forward-looking sonar system. For this reason, the repository includes ARIS parsing utilities under `src/debris_flow/aris/`, adapted from `pyARIS`, to support reading and projecting sonar data during preprocessing.

## Repository Layout

```text
.
|-- src/debris_flow/          # Reusable processing modules
|-- scripts/                  # Command-line entry points
|-- data/                     # Local datasets, not committed
|-- docs/                     # Workflow and hardware notes
|-- requirements.txt
`-- pyproject.toml
```

## Installation

Python 3.8 or newer is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Data Layout

Place image sequences under `data/` or pass absolute paths on the command line. A typical local layout is:

```text
data/
|-- sonar_frames/
|   |-- 000001.jpg
|   |-- 000002.jpg
|   `-- ...
`-- aris/
    `-- sample.aris
```

## Notes

- The tracking workflow estimates dense optical flow between adjacent sonar frames and visualizes the motion field as debris motion vectors.
- Motion vectors are used to analyze the movement behavior of floating debris over time.
- The threshold parameters used in motion estimation may require adaptive tuning under different underwater environments to account for acoustic noise, reverberation, and sonar imaging artifacts.
- The same motion information can be converted into pseudo-label bounding boxes for downstream detection experiments.
- ARIS parsing utilities are included under `src/debris_flow/aris/` for working with `.aris` sonar files.

## Related Repository

- The ARIS data parsing utilities in this project were developed with reference to the public [`pyARIS`](https://github.com/EminentCodfish/pyARIS) repository by Chris Rillahan. 
- Further technical information about ARIS/DIDSON sonar systems is available from the official Sound Metrics GitHub repository: https://github.com/SoundMetrics
