# Gaming

Native Quillcut pipeline. Gameplay, cinematics and streams, with or without commentary.

## How it clips (P0)

1. Ingest: probe, trim baked-in letterbox (cropdetect vote), reject real pictures under 480 px.
2. Analysis (cached per source): audio loudness rise + spectral flux, frame motion + flash +
   histogram cuts at 4 fps, fused into one 0–1 intensity curve at 20 Hz. Silero VAD gives
   speech coverage; ≥ 25 % → `commentary` mode (word captions via faster-whisper).
3. Peaks: greedy non-max suppression on the intensity curve, 45 s apart.
4. Plan: a script template (`scripts/*.yaml`) places footage relative to the peak.
   `quillcut run` uses `cold_open`; `quillcut script <name>` uses one of the 8 STONE XD scripts.
5. Render: per-segment intermediates → concat → captions (libass) → two-pass loudnorm → x264 CRF 18.
6. QC: resolution, frame rate, duration, 30 s floor, loudness, true peak, opening black frames,
   letterbox, caption safe area, unplanned freezes, rights. Any FAIL keeps the clip out of `final/`.

Layout: `center_crop` when a full-height 9:16 crop stays ≤ 2.0× upscale (1080p sources),
`wide_fill` (4:5 window over a blurred copy) otherwise (720p sources). Never black bars.

## Duration policy

The user requires a 30-second minimum on every clip.

> The reference data points the other way. In the gaming corpus the top clip was 16 s
> (140M views) and the second 19 s (113M); outliers centred at 14–30 s. A 30-second floor
> means some moments run past their natural end. This is a product decision, not a
> data-driven one. It is a single config value so it can be A/B tested on our own channels.

Templates reach 30 s with real footage (cold open + setup + payoff). Holds are allowed only
as a deliberate final-frame hold of at most 2 s (scripts 1, 3, 5, 6, 7, 8 ask for one).

## Script templates

`quillcut scripts -v` lists them. Comparison scripts (1, 5, 7, 8) need two sources, `--a` and
`--b`. Use `--a-range 120-300` / `--a-peak 184.5` to point the template at the right moment.

## Known P0 limits

- Peaks come from signals only; the VLM tier (P1) will judge what is actually on screen.
- No music bed yet (`assets/music/manifest.yaml` is empty), so "cut on the beat" is not applied.
- Text hooks come from the script templates; generated, grounded hooks are P1.
