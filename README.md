# Pail AI

This project provides a FastAPI backend for various AI features. A new **video generation mode** is available both in the API and the UI.
The `/video/generate` endpoint accepts a text description and returns a stubbed video URL.
In the web UI a global mode selector allows choosing "ペイルの像(動画生成モード)" to open a dedicated form where you can specify duration, resolution, BGM and more.
The selector now stays visible even when switching to video mode so you can instantly jump back to another mode without closing the form.
Future development will integrate Stable Diffusion, Stable Video Diffusion and other APIs to automatically compose images, BGM and narration into a final video.

## AI Persona

All responses are written in the gentle, sleepy style of **ウキヨザル**, a laid-back monkey character. The AI refers to itself as "ウキヨザル" and calls users "きみ" or "◯◯ちゃん" while speaking slowly with phrases like "〜だね" and "〜かな". Pauses such as "……" or "えっとね" should appear only occasionally—around thirty percent of the time—so that the tone feels natural. Avoid repeating the same endings or excessive self‑introductions. No matter the mode, the output keeps this relaxed style.

## Talk Mode (Temporarily Removed)

The experimental talk mode has been removed from the main application. All `/talk` API endpoints and related UI have been disabled, and dependencies were commented out. Code and data remain archived outside this repository for possible future restoration.

## Registration Flow

Submitting the registration form now navigates to a dedicated progress page. The page shows a large
"登録中..." message with a spinning icon while the server creates the user's profile image in a
background task. Once the file appears the browser automatically redirects to
`profile_generated.html?uid=<id>` where the resulting PNG is displayed fullscreen with a button to
return to the login form.

## Code Generation Mode

For details on output in the `ペイルの創造` code mode, see
[CODE_MODE_OUTPUT_GUIDELINES.md](CODE_MODE_OUTPUT_GUIDELINES.md).

## Logging

Set the `LOG_LEVEL` environment variable to control how verbose the
application logs are. Supported values are the standard Python logging levels
such as `DEBUG`, `INFO` and `WARNING`. If not set, `INFO` is used.

## Installation

The application targets **Python 3.11+**. Create a virtual environment with a
compatible interpreter and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some features rely on system packages. Ensure the `pandoc` command line tool is
available for PDF processing. On Debian/Ubuntu this can be installed with
`apt-get install pandoc`.

The automated browser tests require Playwright's browser binaries. After
installing the Python package, run:

```bash
npx playwright install
```

or `playwright install` if the `playwright` command is on your PATH.

## Configuration

Create a `.env` file based on `.env.example` and provide your API keys. At least
`OPENAI_API_KEY` must be set to enable core features. Keys for Anthropic,
Gemini, Cohere, Perplexity, ElevenLabs and AWS are optional depending on which
functions you plan to use.

To start the development server run:

```bash
uvicorn main:app --reload
```
