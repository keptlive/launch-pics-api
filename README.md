# launch.pics Image Processing API

REST API with 45+ image processing tools. Resize, crop, filter, watermark, compose pipelines, and generate AI-powered workflows.

**Live at:** [launch.pics](https://launch.pics) | **API Docs:** [launch.pics/docs](https://launch.pics/docs/)

## Features

- **45+ image tools** — resize, crop, blur, sharpen, watermark, color adjust, effects, and more
- **Pipelines** — chain up to 20 tools in a single request
- **AI workflows** — describe transformations in plain English, get a ready-to-run pipeline
- **Batch processing** — process up to 10 images in one call
- **Dual input** — accepts file uploads (multipart) or base64 JSON
- **Dual output** — returns base64 JSON or binary image (via Accept header)

## Quick Start

```bash
# Health check (no auth)
curl https://launch.pics/api/v1/health

# List all tools (no auth)
curl https://launch.pics/api/v1/tools

# Resize an image
curl -X POST https://launch.pics/api/v1/resize \
  -H "X-API-Key: lp_your_key" \
  -F "image=@photo.jpg" \
  -F "width=800"

# Run a pipeline
curl -X POST https://launch.pics/api/v1/pipeline \
  -H "X-API-Key: lp_your_key" \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "...",
    "steps": [
      {"tool": "resize", "params": {"width": 800}},
      {"tool": "sepia", "params": {"intensity": 0.6}},
      {"tool": "compress", "params": {"quality": 85, "format": "JPEG"}}
    ]
  }'

# AI-generated workflow
curl -X POST https://launch.pics/api/v1/generate-workflow \
  -H "X-API-Key: lp_your_key" \
  -H "Content-Type: application/json" \
  -d '{"description": "Make it thumbnail-sized with a vintage look"}'
```

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/api/v1/health` | No | Health check and endpoint discovery |
| GET | `/api/v1/tools` | No | List all tools with parameters |
| GET | `/api/v1/fonts` | No | List available fonts for text tools |
| POST | `/api/v1/{tool}` | Yes | Process image with a single tool |
| POST | `/api/v1/pipeline` | Yes | Multi-step pipeline processing |
| POST | `/api/v1/generate-workflow` | Yes | AI-powered workflow generation |
| POST | `/api/v1/batch` | Yes | Batch process multiple images |
| GET | `/api/v1/keys/usage` | Yes | Check API key usage |

## Available Tools

**Image Manipulation:** resize, crop, rotate, mirror, circle-crop, border

**Color & Adjustment:** brightness, contrast, saturation, hue-shift, levels, color-adjust, color-replace, grayscale, invert

**Effects & Filters:** blur, sharpen, sepia, vignette, film-grain, pixelate, posterize, threshold, edge-detect, emboss, solarize, duotone, halftone, oil-paint, sketch, tilt-shift, noise

**Overlay & Compositing:** text-overlay, watermark, shadow, transparency

**Creative & AI:** meme, og-image, gradient-text, badge, vintage, collage, split-tone, color-grade, smart-resize

**Format & Output:** convert, compress

## Font System

45+ fonts available via the `/api/v1/fonts` endpoint. Use any font name in text tools via the `font` parameter.

**Categories:** sans-serif (Roboto, Open Sans, Lato, Ubuntu, Noto Sans), serif (Noto Serif, Liberation Serif), monospace (Fira Code, Ubuntu Mono), display (Roboto Black, Lato Heavy)

**Aliases:** `sans`, `serif`, `mono`, `bold`, `impact`

## Self-Hosting

```bash
# Install dependencies
pip install flask flask-cors flask-limiter pillow

# Set environment variables
export OPENROUTER_API_KEY=your_key    # For AI workflow generation
export LP_ADMIN_SECRET=your_secret     # For API key management

# Run
python image-api.py
# → Server starts on port 3240
```

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| Single tool | 200/hour |
| Pipeline | 100/hour |
| Batch | 50/hour |
| AI workflow | 30/hour |

## License

MIT
