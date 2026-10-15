<p align="center">
  <img src="src/quillcut/web/static/assets/banner.png" alt="Quillcut Studio" width="720">
</p>

# Quillcut Studio

Drop in one long video of people talking. Quillcut reads the **whole conversation first**, picks the
moments that are complete (a question and its full answer, a finished story, a finished explanation),
edits each one properly, and gives you reels with a caption and hashtags, ready to download.

Everything runs on your own computer. Nothing is uploaded anywhere.

## What it does differently

| Problem with typical clippers | What Quillcut does |
|---|---|
| Cuts at a timestamp, mid-answer | Never starts or ends mid-sentence. Reels end where the thought ends. |
| Fixed length for every clip | Each reel has its own target (30, 35, 60, 90… or any value). The target is a goal: a reel runs longer when the answer needs it, and tells you so. |
| Ignores who is speaking | Separates the interviewer from the guest and keeps the question with its answer. |
| Subtitles and nothing else | Speaker-following framing, punch-ins, an animated title, key-term emphasis, key-phrase callouts, a lower-third and loudness-matched audio. |
| Makes filler to hit the count | If the video only supports 3 distinct complete moments and you ask for 5, you get 3 and an explanation. |

```mermaid
flowchart LR
    A["1. Drop your video"] --> B["2. Quillcut reads the whole<br/>conversation and reports<br/>how many complete moments exist"]
    B --> C["3. Choose reel count and a<br/>length for each reel"]
    C --> D["4. Pick a look and<br/>editing intensity"]
    D --> E["5. Download reels with<br/>hook, caption, hashtags"]
```

## Quick start (Windows 10/11)

You need Python 3.11 and [ffmpeg](https://ffmpeg.org) on your PATH. An NVIDIA GPU and
[Ollama](https://ollama.com) are optional but recommended.

```powershell
git clone https://github.com/VARSHITH707/quillcut.git
cd quillcut
.\launch.bat
```

The first run sets everything up (a few GB, several minutes) and opens the app at
<http://127.0.0.1:8765>. After that it starts in seconds. Keep the window open while you use it.

For AI-written captions and hashtags (optional, free, runs locally):

```powershell
ollama pull qwen2.5:7b-instruct-q4_K_M
```

<details>
<summary>Manual setup instead of launch.bat</summary>

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
# NVIDIA GPU (CUDA 12.6)
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126
# or CPU only
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
quillcut web
```
</details>

## Using the app

1. **Upload** your video (MP4, MOV, MKV, WEBM; at least 20 seconds; 720p or higher works best).
   Quillcut immediately reads the whole thing and tells you how many distinct, complete moments it
   found, with the opening line of each.
2. **Reels:** choose how many (1, 2, 3, 5, 10 or any number up to 20). One duration control appears per
   reel, with presets of 30, 35, 45, 60, 75 and 90 seconds plus a custom value.
3. **Look:** caption style, content style (interview, podcast, educational, storytelling, motivational,
   debate, business), editing intensity, and aspect ratio.
4. **Make it yours:** caption size and position, colours, brand colour, optional logo, closing line,
   optional quiet background music.
5. **Confirm** you have the rights, and make the reels.

Each finished reel shows the **target and the actual duration**, whether it was **extended to finish the
thought** (and by how much), the **hook line** it opens on, the **source timestamps**, the **checks it
passed**, an editable **caption and hashtags**, and download buttons. Two buttons let you try again:
*New length* re-cuts the same moment to a different target, and *Try another moment* picks the next-best one.

### Duration rules

* The selected duration is a **target**. If the speaker is still finishing the point at the target, the
  reel is extended to the natural end, up to a hard maximum (about 1.75× the target, at least +30 s).
* A reel is never padded. If the thought finishes early, it ends early and says so.
* **Strict mode** never runs past the target. If a complete answer does not fit, that reel is skipped and
  you are told how long the answer needs, instead of cutting it off.
* A 90-second answer that is longer than the maximum ends at a paragraph boundary and is marked as a part.

## How the clipping decision is made

```mermaid
flowchart TD
    A[Transcribe every word with timestamps] --> B[Split into sentences,<br/>mark questions and unfinished thoughts]
    B --> C[Separate voices<br/>interviewer vs guest]
    C --> D[Group into units:<br/>question + answer, or a talk]
    D --> E[Split units into paragraphs<br/>where the topic shifts]
    E --> F[Build candidates from whole paragraphs<br/>hard gates: complete, self-contained]
    F --> G[Rank: completeness, fit to target,<br/>hook, information, clarity, novelty]
    G --> H[Pick distinct moments<br/>one per requested reel]
    H --> I[Optional second opinion from<br/>the local language model]
```

A candidate must pass these gates before it is ever ranked: it starts on a sentence start, ends on a
finished sentence, is not continued by the next sentence, does not open on a reference like "so" or
"that" without its question, and (for a question and answer) reaches the end of the answer unless the
answer exceeds the maximum.

## The edit

* **Hook (first 3 seconds):** a verbatim line from the conversation, shown in one or two frames of at most
  five words. It is the question, the opening line, or a stronger line from later in the same answer played
  first as a cold open. Each hook is labelled with its archetype (loss avoidance, insider information, status
  and optimisation, contrarian) when the speaker's own words fit one. Nothing is rewritten or invented, and a
  shortened line is marked with "…".
* **Framing:** vertical crops follow the active speaker (face detection, with mouth movement choosing
  between several faces). Framing changes happen on sentence boundaries.
* **Pacing:** silences longer than 0.40 s are tightened to 0.18 s, and filler sounds ("um", "uh") are removed.
  Real words are never removed, including "like" and "you know".
* **Motion graphics:** animated title with an accent bar, a "Question" lower-third, key terms in the
  accent colour, key-phrase callouts quoted from the speaker, and an optional progress bar and closing line.
* **Sound:** loudness matched, peaks limited, optional noise reduction, optional music that ducks under speech.
* **Intensity** (a framing change falls between two words, so audio runs straight through every cut):
  *clean* every 3-4 s at 1.0x/1.05x; *dynamic* every 2-2.5 s at 1.0x/1.10x/1.14x with a callout;
  *high energy* every 1.2-1.8 s at 1.0x/1.12x/1.18x with two callouts and a progress bar.
* **Captions:** 3-5 words per card, locked to the centre third of the frame (35%-65% of the height; 60% by
  default), active word highlighted, bold type with a dark outline.
* **Post copy:** a re-hook under 10 words, 3-4 bullets taken from the speaker's own sentences, and a call to
  action by clip type (educational: "Save this guide for later.", opinion or debate: "Comment your thoughts
  below.", business: "Send this to a teammate."), plus hashtags.

Every reel is **audited after rendering**: the speech is transcribed again and compared with the plan, the hook
must be on screen in the first second, captions must be 5 words or fewer and inside the centre third, and every
overlay must sit inside the safe area. A reel that fails is flagged "needs a look" instead of silently delivered.

## Command line

```powershell
quillcut clip interview.mp4 --targets 35,60,90 --rights own_footage --intensity dynamic
quillcut web                                  # the app
quillcut run match.mp4 --format gaming        # the original loudness-peak clipper
quillcut doctor                               # check ffmpeg, GPU, fonts and models
```

| `clip` option | Meaning |
|---|---|
| `--targets 35,60,90` | One target (seconds) per reel |
| `--strict` | Never exceed a target; report the conflict instead |
| `--include-question auto\|yes\|no` | Keep the interviewer's question in the reel |
| `--intensity clean\|dynamic\|high_energy` | How much editing |
| `--aspect 9:16\|1:1\|16:9` | Output shape |
| `--content-style interview\|podcast\|…` | Ranking bias for the kind of content |
| `--no-llm`, `--no-verify` | Skip the model second opinion / the re-listen check |

## Honest limits

* **Windows only** today (the GPU lock and caption fonts are Windows-specific).
* **Speaker separation** is an estimate from voice pitch and tone. It works well for two clearly
  different voices; very similar voices may be treated as one. It does not name people.
* **Face-following framing** is tested on synthetic detections, not on a library of real interview
  footage. It falls back to a centred crop when no face is found.
* **The language model is advisory.** A small local model is a noisy judge, so it can only veto a moment
  when a separate deterministic check also says the speaker is still on the same point.
* **Music** needs a track on disk. Run `python tools/make_music.py` to generate royalty-free beds.
* The video must contain **speech**. Gameplay without commentary has no conversation to clip.

## Project layout

```text
quillcut/
├── src/quillcut/
│   ├── conversation.py      # sentences, units, candidates, ranking, selection (pure logic)
│   ├── speakers.py          # voice separation
│   ├── reelplan.py          # moment -> edit plan (hook, cuts, framing, graphics)
│   ├── reframe.py           # follow the active speaker
│   ├── conversation_job.py  # whole pipeline + post-render verification
│   ├── render.py, captions.py, qc.py   # rendering, motion graphics, quality checks
│   └── web/                 # local app: server, job types, static page
├── formats/  config/  tests/  tools/  demo/
```

## Tests and security

```powershell
pytest tests/unit -q          # fast: engine, planner, server (about 30 s)
pytest tests -q               # everything, including real-speech acceptance tests (several minutes)
bandit -r src tools           # static security scan
```

The acceptance tests synthesize a two-voice interview with Windows text-to-speech, run the full
pipeline, and check the rendered files: complete answers, extended durations, voice separation, and
honest shortfalls.

## License

[MIT](LICENSE) © 2026 VARSHITH707

