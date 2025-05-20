# Deference Image Variation and Talk Mode Notes

This repository now includes a `deference` setting for image generation.
It ranges from **1** (almost identical images) to **5** (maximal variation).
The API `POST /images/generate` accepts the following additional fields:

- `deference` (int 1-5, default 3)
- `allow_text` (bool, default True)
- `text_content` (optional string)

When `allow_text` is False, the backend adds negative prompts to avoid any
letters or watermarks. When `text_content` is provided, it is appended to
the prompt so the requested text is drawn in the image as accurately as
possible.

The experimental `talk` router has been removed from the main application.
Archived copies of the code remain available outside this repository for
potential future integration.
