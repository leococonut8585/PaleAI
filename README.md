# Pail AI

This project provides a FastAPI backend for various AI features. A new **video generation mode** is available both in the API and the UI.
The `/video/generate` endpoint accepts a text description and returns a stubbed video URL.
In the web UI a global mode selector allows choosing "動画生成" to open a dedicated form where you can specify duration, resolution, BGM and more.
The selector now stays visible even when switching to video mode so you can instantly jump back to another mode without closing the form.
Future development will integrate Stable Diffusion, Stable Video Diffusion and other APIs to automatically compose images, BGM and narration into a final video.

## Talk Mode (Temporarily Removed)

The experimental talk mode has been removed from the main application. All `/talk` API endpoints and related UI have been disabled, and dependencies were commented out. Code and data remain archived outside this repository for possible future restoration.

## Registration Flow

Submitting the registration form now navigates to a dedicated progress page. The page shows a large
"登録中..." message with a spinning icon while the server creates the user's profile image in a
background task. Once the file appears the browser automatically redirects to
`profile_generated.html?uid=<id>` where the resulting PNG is displayed fullscreen with a button to
return to the login form.
