#!/usr/bin/env python3
"""launch.pics Image Processing API — REST endpoints for all image tools."""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont, ImageOps
import io
import base64
import json
import os
import hashlib
import time
import secrets
from functools import wraps
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
CORS(app)

MAX_SIZE = 20 * 1024 * 1024  # 20MB max upload

def get_image_from_request():
    """Extract image from request (file upload or base64)."""
    if 'image' in request.files:
        f = request.files['image']
        return Image.open(f.stream)
    elif request.is_json and request.get_json(silent=True) and 'image_base64' in request.get_json(silent=True):
        data = base64.b64decode(request.get_json(silent=True)['image_base64'])
        return Image.open(io.BytesIO(data))
    return None

def image_to_response(img, fmt='PNG', quality=90):
    """Convert PIL Image to response."""
    buf = io.BytesIO()
    save_kwargs = {'format': fmt}
    if fmt.upper() in ('JPEG', 'WEBP'):
        save_kwargs['quality'] = quality
    if img.mode == 'RGBA' and fmt.upper() == 'JPEG':
        img = img.convert('RGB')
    img.save(buf, **save_kwargs)
    buf.seek(0)
    mime = {'PNG': 'image/png', 'JPEG': 'image/jpeg', 'WEBP': 'image/webp', 'GIF': 'image/gif'}
    return send_file(buf, mimetype=mime.get(fmt.upper(), 'image/png'))

def image_to_base64(img, fmt='PNG', quality=90):
    """Convert PIL Image to base64 string."""
    buf = io.BytesIO()
    save_kwargs = {'format': fmt}
    if fmt.upper() in ('JPEG', 'WEBP'):
        save_kwargs['quality'] = quality
    if img.mode == 'RGBA' and fmt.upper() == 'JPEG':
        img = img.convert('RGB')
    img.save(buf, **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode()

def error(msg, code=400):
    return jsonify({'error': msg}), code

def get_params():
    """Get request parameters from form data or JSON body, safely."""
    if request.form:
        return request.form
    return request.get_json(silent=True) or {}


# ============================================================
# API ENDPOINTS
# ============================================================

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'version': '1.0', 'tools': list(TOOL_REGISTRY.keys())})


@app.route('/api/tools', methods=['GET'])
def list_tools():
    """List all available tools with their parameters."""
    return jsonify({'tools': TOOL_REGISTRY})


# --- RESIZE ---
@app.route('/api/resize', methods=['POST'])
def resize():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    width = request.form.get('width') or (request.json or {}).get('width')
    height = request.form.get('height') or (request.json or {}).get('height')
    mode = request.form.get('mode', 'contain') or (request.json or {}).get('mode', 'contain')

    if not width and not height:
        return error('Provide width and/or height')

    w, h = img.size
    width = int(width) if width else None
    height = int(height) if height else None

    if mode == 'stretch':
        if not width: width = w
        if not height: height = h
        img = img.resize((width, height), Image.LANCZOS)
    elif mode == 'cover':
        if not width: width = w
        if not height: height = h
        img = ImageOps.fit(img, (width, height), Image.LANCZOS)
    else:  # contain
        if width and height:
            img.thumbnail((width, height), Image.LANCZOS)
        elif width:
            ratio = width / w
            img = img.resize((width, int(h * ratio)), Image.LANCZOS)
        else:
            ratio = height / h
            img = img.resize((int(w * ratio), height), Image.LANCZOS)

    fmt = request.form.get('format', 'PNG') or (request.json or {}).get('format', 'PNG')
    return image_to_response(img, fmt.upper())


# --- CROP ---
@app.route('/api/crop', methods=['POST'])
def crop():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    x = int(data.get('x', 0))
    y = int(data.get('y', 0))
    width = int(data.get('width', img.size[0]))
    height = int(data.get('height', img.size[1]))

    img = img.crop((x, y, x + width, y + height))
    return image_to_response(img)


# --- ROTATE ---
@app.route('/api/rotate', methods=['POST'])
def rotate():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    angle = float(data.get('angle', 90))
    expand = data.get('expand', 'true').lower() == 'true'
    flip_h = data.get('flip_h', 'false').lower() == 'true'
    flip_v = data.get('flip_v', 'false').lower() == 'true'

    if flip_h:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if flip_v:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if angle:
        img = img.rotate(-angle, expand=expand, resample=Image.BICUBIC)

    return image_to_response(img)


# --- BLUR ---
@app.route('/api/blur', methods=['POST'])
def blur():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    radius = float(data.get('radius', 5))
    blur_type = data.get('type', 'gaussian')

    if blur_type == 'box':
        img = img.filter(ImageFilter.BoxBlur(radius))
    else:
        img = img.filter(ImageFilter.GaussianBlur(radius))

    return image_to_response(img)


# --- SHARPEN ---
@app.route('/api/sharpen', methods=['POST'])
def sharpen():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    amount = float(data.get('amount', 2.0))

    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(amount)
    return image_to_response(img)


# --- BRIGHTNESS ---
@app.route('/api/brightness', methods=['POST'])
def brightness():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    factor = float(data.get('factor', 1.2))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(factor)
    return image_to_response(img)


# --- CONTRAST ---
@app.route('/api/contrast', methods=['POST'])
def contrast():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    factor = float(data.get('factor', 1.5))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(factor)
    return image_to_response(img)


# --- SATURATION ---
@app.route('/api/saturation', methods=['POST'])
def saturation():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    factor = float(data.get('factor', 1.5))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(factor)
    return image_to_response(img)


# --- GRAYSCALE ---
@app.route('/api/grayscale', methods=['POST'])
def grayscale():
    img = get_image_from_request()
    if not img:
        return error('No image provided')

    img = img.convert('L').convert('RGB')
    return image_to_response(img)


# --- SEPIA ---
@app.route('/api/sepia', methods=['POST'])
def sepia():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    intensity = float(data.get('intensity', 0.8))

    gray = img.convert('L')
    w, h = gray.size
    sepia_img = Image.new('RGB', (w, h))
    pixels = gray.load()
    sepia_pixels = sepia_img.load()

    for y in range(h):
        for x in range(w):
            g = pixels[x, y]
            r = min(255, int(g + 94 * intensity))
            gn = min(255, int(g + 38 * intensity))
            b = max(0, int(g - 20 * intensity))
            sepia_pixels[x, y] = (r, gn, b)

    return image_to_response(sepia_img)


# --- INVERT ---
@app.route('/api/invert', methods=['POST'])
def invert():
    img = get_image_from_request()
    if not img:
        return error('No image provided')

    if img.mode == 'RGBA':
        r, g, b, a = img.split()
        rgb = Image.merge('RGB', (r, g, b))
        rgb = ImageOps.invert(rgb)
        r2, g2, b2 = rgb.split()
        img = Image.merge('RGBA', (r2, g2, b2, a))
    else:
        img = ImageOps.invert(img.convert('RGB'))

    return image_to_response(img)


# --- FORMAT CONVERT ---
@app.route('/api/convert', methods=['POST'])
def convert_format():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    fmt = data.get('format', 'PNG').upper()
    quality = int(data.get('quality', 90))

    if fmt not in ('PNG', 'JPEG', 'WEBP', 'GIF'):
        return error('Unsupported format. Use PNG, JPEG, WEBP, or GIF')

    return image_to_response(img, fmt, quality)


# --- COMPRESS ---
@app.route('/api/compress', methods=['POST'])
def compress():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    quality = int(data.get('quality', 60))
    fmt = data.get('format', 'JPEG').upper()

    return image_to_response(img, fmt, quality)


# --- WATERMARK ---
@app.route('/api/watermark', methods=['POST'])
def watermark():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    text = data.get('text', 'launch.pics')
    opacity = int(data.get('opacity', 128))
    position = data.get('position', 'bottom-right')
    font_size = int(data.get('font_size', 24))

    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', font_size)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    w, h = img.size
    padding = 20

    positions = {
        'top-left': (padding, padding),
        'top-right': (w - tw - padding, padding),
        'bottom-left': (padding, h - th - padding),
        'bottom-right': (w - tw - padding, h - th - padding),
        'center': ((w - tw) // 2, (h - th) // 2),
    }
    pos = positions.get(position, positions['bottom-right'])

    draw.text(pos, text, fill=(255, 255, 255, opacity), font=font)

    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    img = Image.alpha_composite(img, overlay)

    return image_to_response(img.convert('RGB'), 'PNG')


# --- EDGE DETECTION ---
@app.route('/api/edge-detect', methods=['POST'])
def edge_detect():
    img = get_image_from_request()
    if not img:
        return error('No image provided')

    img = img.filter(ImageFilter.FIND_EDGES)
    return image_to_response(img)


# --- EMBOSS ---
@app.route('/api/emboss', methods=['POST'])
def emboss():
    img = get_image_from_request()
    if not img:
        return error('No image provided')

    img = img.filter(ImageFilter.EMBOSS)
    return image_to_response(img)


# --- POSTERIZE ---
@app.route('/api/posterize', methods=['POST'])
def posterize():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    bits = int(data.get('bits', 4))
    bits = max(1, min(8, bits))

    img = ImageOps.posterize(img.convert('RGB'), bits)
    return image_to_response(img)


# --- PIXELATE ---
@app.route('/api/pixelate', methods=['POST'])
def pixelate():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    block_size = int(data.get('block_size', 10))

    w, h = img.size
    small = img.resize((max(1, w // block_size), max(1, h // block_size)), Image.NEAREST)
    img = small.resize((w, h), Image.NEAREST)
    return image_to_response(img)


# --- THRESHOLD ---
@app.route('/api/threshold', methods=['POST'])
def threshold():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    level = int(data.get('level', 128))

    img = img.convert('L').point(lambda x: 255 if x > level else 0, '1')
    return image_to_response(img.convert('RGB'))


# --- VIGNETTE ---
@app.route('/api/vignette', methods=['POST'])
def vignette():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    strength = float(data.get('strength', 0.5))

    w, h = img.size
    mask = Image.new('L', (w, h), 255)
    draw = ImageDraw.Draw(mask)

    for i in range(min(w, h) // 2):
        alpha = int(255 * (1 - strength * (1 - i / (min(w, h) / 2)) ** 2))
        alpha = max(0, min(255, alpha))
        draw.ellipse([i, i, w - i, h - i], fill=alpha)

    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    dark = Image.new('RGBA', (w, h), (0, 0, 0, 255))
    img = Image.composite(img, dark, mask)
    return image_to_response(img.convert('RGB'))


# --- FILM GRAIN ---
@app.route('/api/film-grain', methods=['POST'])
def film_grain():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    amount = float(data.get('amount', 0.3))

    import random
    w, h = img.size
    pixels = img.load()

    for y in range(h):
        for x in range(w):
            noise = int((random.random() - 0.5) * 255 * amount)
            p = pixels[x, y]
            if isinstance(p, int):
                pixels[x, y] = max(0, min(255, p + noise))
            else:
                pixels[x, y] = tuple(max(0, min(255, c + noise)) for c in p[:3])

    return image_to_response(img)


# --- SHADOW ---
@app.route('/api/shadow', methods=['POST'])
def shadow():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    offset_x = int(data.get('offset_x', 10))
    offset_y = int(data.get('offset_y', 10))
    blur_radius = int(data.get('blur', 15))
    color = data.get('color', '#000000')

    w, h = img.size
    padding = max(abs(offset_x), abs(offset_y)) + blur_radius * 2
    new_w, new_h = w + padding * 2, h + padding * 2

    shadow_img = Image.new('RGBA', (new_w, new_h), (0, 0, 0, 0))
    shadow_layer = Image.new('RGBA', (w, h), color + 'AA')
    shadow_img.paste(shadow_layer, (padding + offset_x, padding + offset_y))
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(blur_radius))

    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    shadow_img.paste(img, (padding, padding), img)

    return image_to_response(shadow_img)


# --- COLOR ADJUST (temperature/tint) ---
@app.route('/api/color-adjust', methods=['POST'])
def color_adjust():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    temperature = float(data.get('temperature', 0))  # -100 to 100
    tint = float(data.get('tint', 0))  # -100 to 100

    img = img.convert('RGB')
    pixels = img.load()
    w, h = img.size

    for y in range(h):
        for x in range(w):
            r, g, b = pixels[x, y]
            r = max(0, min(255, int(r + temperature * 0.5)))
            b = max(0, min(255, int(b - temperature * 0.5)))
            g = max(0, min(255, int(g + tint * 0.3)))
            pixels[x, y] = (r, g, b)

    return image_to_response(img)


# --- HUE SHIFT ---
@app.route('/api/hue-shift', methods=['POST'])
def hue_shift():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    degrees = float(data.get('degrees', 30))

    img = img.convert('HSV')
    pixels = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            h_val, s, v = pixels[x, y]
            h_val = int((h_val + degrees * 255 / 360) % 256)
            pixels[x, y] = (h_val, s, v)
    img = img.convert('RGB')
    return image_to_response(img)


# --- DUOTONE ---
@app.route('/api/duotone', methods=['POST'])
def duotone():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    dark = data.get('dark_color', '#001848')
    light = data.get('light_color', '#f7a800')

    def hex_to_rgb(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    d = hex_to_rgb(dark)
    l = hex_to_rgb(light)
    gray = img.convert('L')
    w, h_px = gray.size
    out = Image.new('RGB', (w, h_px))
    gp = gray.load()
    op = out.load()
    for y in range(h_px):
        for x in range(w):
            t = gp[x, y] / 255.0
            op[x, y] = tuple(int(d[i] * (1 - t) + l[i] * t) for i in range(3))
    return image_to_response(out)


# --- HALFTONE ---
@app.route('/api/halftone', methods=['POST'])
def halftone():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    dot_size = int(data.get('dot_size', 8))

    gray = img.convert('L')
    w, h = gray.size
    out = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    gp = gray.load()

    for y in range(0, h, dot_size):
        for x in range(0, w, dot_size):
            block_sum = 0
            count = 0
            for dy in range(min(dot_size, h - y)):
                for dx in range(min(dot_size, w - x)):
                    block_sum += gp[x + dx, y + dy]
                    count += 1
            avg = block_sum / count if count else 128
            radius = int((1 - avg / 255) * dot_size / 2)
            if radius > 0:
                cx, cy = x + dot_size // 2, y + dot_size // 2
                draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(0, 0, 0))

    return image_to_response(out)


# --- OIL PAINT ---
@app.route('/api/oil-paint', methods=['POST'])
def oil_paint():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    radius = int(data.get('radius', 4))

    img = img.convert('RGB')
    w, h = img.size
    src = img.load()
    out = img.copy()
    dst = out.load()

    for y in range(radius, h - radius):
        for x in range(radius, w - radius):
            intensities = {}
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    r, g, b = src[x + dx, y + dy]
                    intensity = (r + g + b) // 3 // 32
                    if intensity not in intensities:
                        intensities[intensity] = [0, 0, 0, 0]
                    intensities[intensity][0] += r
                    intensities[intensity][1] += g
                    intensities[intensity][2] += b
                    intensities[intensity][3] += 1
            max_count = max(intensities.values(), key=lambda x: x[3])
            c = max_count[3]
            dst[x, y] = (max_count[0] // c, max_count[1] // c, max_count[2] // c)

    return image_to_response(out)


# --- SKETCH ---
@app.route('/api/sketch', methods=['POST'])
def sketch():
    img = get_image_from_request()
    if not img:
        return error('No image provided')

    gray = img.convert('L')
    inverted = ImageOps.invert(gray)
    blurred = inverted.filter(ImageFilter.GaussianBlur(21))

    w, h = gray.size
    gp = gray.load()
    bp = blurred.load()
    out = Image.new('L', (w, h))
    op = out.load()
    for y in range(h):
        for x in range(w):
            b_val = bp[x, y]
            if b_val == 0:
                op[x, y] = 0
            else:
                val = min(255, int(gp[x, y] * 256 / (256 - b_val + 1)))
                op[x, y] = val

    return image_to_response(out.convert('RGB'))


# --- SOLARIZE ---
@app.route('/api/solarize', methods=['POST'])
def solarize():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    threshold_val = int(data.get('threshold', 128))

    img = ImageOps.solarize(img.convert('RGB'), threshold_val)
    return image_to_response(img)


# --- MIRROR ---
@app.route('/api/mirror', methods=['POST'])
def mirror():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    direction = data.get('direction', 'horizontal')

    if direction == 'vertical':
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return image_to_response(img)


# --- TILT SHIFT ---
@app.route('/api/tilt-shift', methods=['POST'])
def tilt_shift():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    blur_amount = float(data.get('blur', 10))
    focus_position = float(data.get('focus_position', 0.5))
    focus_width = float(data.get('focus_width', 0.2))

    w, h = img.size
    blurred = img.filter(ImageFilter.GaussianBlur(blur_amount))

    # Create gradient mask
    mask = Image.new('L', (w, h), 0)
    for y in range(h):
        t = y / h
        dist = abs(t - focus_position) / max(focus_width, 0.01)
        alpha = min(255, int(max(0, dist - 1) * 255))
        for x in range(w):
            mask.putpixel((x, y), alpha)

    img = Image.composite(blurred, img, mask)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.3)
    return image_to_response(img)


# --- CIRCLE CROP ---
@app.route('/api/circle-crop', methods=['POST'])
def circle_crop():
    img = get_image_from_request()
    if not img:
        return error('No image provided')

    w, h = img.size
    size = min(w, h)
    img = ImageOps.fit(img, (size, size), Image.LANCZOS)

    mask = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([0, 0, size, size], fill=255)

    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    out = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    out.paste(img, mask=mask)
    return image_to_response(out)


# --- BORDER / FRAME ---
@app.route('/api/border', methods=['POST'])
def add_border():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    width_val = int(data.get('width', 20))
    color = data.get('color', '#ffffff')

    def hex_to_rgb(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    c = hex_to_rgb(color)
    img = ImageOps.expand(img.convert('RGB'), border=width_val, fill=c)
    return image_to_response(img)


# --- LEVELS ---
@app.route('/api/levels', methods=['POST'])
def levels():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    black_point = int(data.get('black_point', 0))
    white_point = int(data.get('white_point', 255))
    gamma = float(data.get('gamma', 1.0))

    img = img.convert('RGB')
    pixels = img.load()
    w, h = img.size
    rng = max(1, white_point - black_point)

    for y in range(h):
        for x in range(w):
            r, g, b = pixels[x, y]
            channels = []
            for c in (r, g, b):
                c = max(0, min(255, (c - black_point) * 255 // rng))
                c = int(255 * (c / 255.0) ** (1.0 / gamma))
                channels.append(max(0, min(255, c)))
            pixels[x, y] = tuple(channels)

    return image_to_response(img)


# --- COLOR REPLACE ---
@app.route('/api/color-replace', methods=['POST'])
def color_replace():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    from_color = data.get('from_color', '#ff0000')
    to_color = data.get('to_color', '#0000ff')
    tolerance = int(data.get('tolerance', 40))

    def hex_to_rgb(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    fc = hex_to_rgb(from_color)
    tc = hex_to_rgb(to_color)
    img = img.convert('RGB')
    pixels = img.load()
    w, h = img.size

    for y in range(h):
        for x in range(w):
            r, g, b = pixels[x, y]
            dist = ((r - fc[0])**2 + (g - fc[1])**2 + (b - fc[2])**2) ** 0.5
            if dist <= tolerance:
                t = max(0, 1 - dist / tolerance)
                pixels[x, y] = (
                    int(r + (tc[0] - r) * t),
                    int(g + (tc[1] - g) * t),
                    int(b + (tc[2] - b) * t)
                )

    return image_to_response(img)


# --- NOISE ADD ---
@app.route('/api/noise', methods=['POST'])
def noise():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    amount = float(data.get('amount', 0.2))
    noise_type = data.get('type', 'gaussian')

    import random
    img = img.convert('RGB')
    pixels = img.load()
    w, h = img.size

    for y in range(h):
        for x in range(w):
            r, g, b = pixels[x, y]
            if noise_type == 'salt_pepper':
                if random.random() < amount * 0.5:
                    val = 255 if random.random() > 0.5 else 0
                    pixels[x, y] = (val, val, val)
            else:
                n = int(random.gauss(0, amount * 128))
                pixels[x, y] = (max(0, min(255, r+n)), max(0, min(255, g+n)), max(0, min(255, b+n)))

    return image_to_response(img)


# --- TEXT OVERLAY ---
@app.route('/api/text-overlay', methods=['POST'])
def text_overlay():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()
    text = data.get('text', 'Hello')
    font_size = int(data.get('font_size', 48))
    color = data.get('color', '#ffffff')
    opacity = int(data.get('opacity', 255))
    position = data.get('position', 'center')
    x = data.get('x')
    y_pos = data.get('y')
    rotation = float(data.get('rotation', 0))
    stroke_color = data.get('stroke_color', '')
    stroke_width = int(data.get('stroke_width', 0))
    line_spacing = int(data.get('line_spacing', 4))

    def hex_to_rgba(h, a=255):
        h = h.lstrip('#')
        if len(h) == 6:
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) + (a,)
        return (255, 255, 255, a)

    fill = hex_to_rgba(color, opacity)
    stroke = hex_to_rgba(stroke_color, opacity) if stroke_color else None

    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', font_size)
    except:
        font = ImageFont.load_default()

    # Create text layer
    txt_layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(txt_layer)

    # Calculate text size
    lines = text.replace('\\n', '\n').split('\n')
    line_heights = []
    line_widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])
    total_h = sum(line_heights) + line_spacing * (len(lines) - 1)
    max_w = max(line_widths) if line_widths else 0

    w, h = img.size
    pad = 20

    # Determine position
    if x is not None and y_pos is not None:
        tx, ty = int(x), int(y_pos)
    else:
        positions = {
            'top-left': (pad, pad),
            'top-center': ((w - max_w) // 2, pad),
            'top-right': (w - max_w - pad, pad),
            'center-left': (pad, (h - total_h) // 2),
            'center': ((w - max_w) // 2, (h - total_h) // 2),
            'center-right': (w - max_w - pad, (h - total_h) // 2),
            'bottom-left': (pad, h - total_h - pad),
            'bottom-center': ((w - max_w) // 2, h - total_h - pad),
            'bottom-right': (w - max_w - pad, h - total_h - pad),
        }
        tx, ty = positions.get(position, positions['center'])

    # Draw each line
    cy = ty
    for i, line in enumerate(lines):
        kwargs = {'fill': fill, 'font': font}
        if stroke and stroke_width > 0:
            kwargs['stroke_fill'] = stroke
            kwargs['stroke_width'] = stroke_width
        draw.text((tx, cy), line, **kwargs)
        cy += line_heights[i] + line_spacing

    # Rotate text layer if needed
    if rotation:
        txt_layer = txt_layer.rotate(rotation, expand=False, resample=Image.BICUBIC)

    img = Image.alpha_composite(img, txt_layer)
    return image_to_response(img, 'PNG')


# --- LAYER COMPOSITE ---
@app.route('/api/layer-composite', methods=['POST'])
def layer_composite():
    """Composite two images as layers with blend modes and positioning."""
    if not request.is_json:
        # Try multipart with two images
        base_file = request.files.get('base') or request.files.get('image')
        overlay_file = request.files.get('overlay') or request.files.get('layer')
        if not base_file or not overlay_file:
            return error('Provide base and overlay images')
        base_img = Image.open(base_file.stream)
        overlay_img = Image.open(overlay_file.stream)
        data = request.form or {}
    else:
        data = request.json
        if 'base_base64' not in data or 'overlay_base64' not in data:
            return error('Provide base_base64 and overlay_base64')
        base_img = Image.open(io.BytesIO(base64.b64decode(data['base_base64'])))
        overlay_img = Image.open(io.BytesIO(base64.b64decode(data['overlay_base64'])))

    opacity = float(data.get('opacity', 1.0))
    x_offset = int(data.get('x', 0))
    y_offset = int(data.get('y', 0))
    blend_mode = data.get('blend_mode', 'normal')
    resize_overlay = data.get('resize_overlay', '')

    if base_img.mode != 'RGBA':
        base_img = base_img.convert('RGBA')
    if overlay_img.mode != 'RGBA':
        overlay_img = overlay_img.convert('RGBA')

    # Optionally resize overlay
    if resize_overlay == 'fit':
        overlay_img.thumbnail(base_img.size, Image.LANCZOS)
    elif resize_overlay == 'fill':
        overlay_img = ImageOps.fit(overlay_img, base_img.size, Image.LANCZOS)
    elif resize_overlay == 'stretch':
        overlay_img = overlay_img.resize(base_img.size, Image.LANCZOS)

    # Apply opacity
    if opacity < 1.0:
        r, g, b, a = overlay_img.split()
        a = a.point(lambda p: int(p * opacity))
        overlay_img = Image.merge('RGBA', (r, g, b, a))

    # Blend modes
    if blend_mode == 'normal':
        canvas = base_img.copy()
        canvas.paste(overlay_img, (x_offset, y_offset), overlay_img)
        result = canvas
    elif blend_mode == 'multiply':
        from PIL import ImageChops
        # Resize overlay to match base for blending
        ov = Image.new('RGBA', base_img.size, (255, 255, 255, 0))
        ov.paste(overlay_img, (x_offset, y_offset), overlay_img)
        base_rgb = base_img.convert('RGB')
        ov_rgb = ov.convert('RGB')
        blended = ImageChops.multiply(base_rgb, ov_rgb)
        result = blended.convert('RGBA')
        result.putalpha(base_img.split()[3])
    elif blend_mode == 'screen':
        from PIL import ImageChops
        ov = Image.new('RGBA', base_img.size, (0, 0, 0, 0))
        ov.paste(overlay_img, (x_offset, y_offset), overlay_img)
        base_rgb = base_img.convert('RGB')
        ov_rgb = ov.convert('RGB')
        blended = ImageChops.screen(base_rgb, ov_rgb)
        result = blended.convert('RGBA')
        result.putalpha(base_img.split()[3])
    elif blend_mode == 'overlay':
        from PIL import ImageChops
        ov = Image.new('RGBA', base_img.size, (128, 128, 128, 0))
        ov.paste(overlay_img, (x_offset, y_offset), overlay_img)
        base_rgb = base_img.convert('RGB')
        ov_rgb = ov.convert('RGB')
        blended = ImageChops.overlay(base_rgb, ov_rgb) if hasattr(ImageChops, 'overlay') else ImageChops.multiply(base_rgb, ov_rgb)
        result = blended.convert('RGBA')
        result.putalpha(base_img.split()[3])
    else:
        canvas = base_img.copy()
        canvas.paste(overlay_img, (x_offset, y_offset), overlay_img)
        result = canvas

    return image_to_response(result, 'PNG')


# --- SET TRANSPARENCY ---
@app.route('/api/transparency', methods=['POST'])
def transparency():
    img = get_image_from_request()
    if not img:
        return error('No image provided')
    data = get_params()

    opacity = data.get('opacity')
    remove_color = data.get('remove_color', '')
    tolerance = int(data.get('tolerance', 30))

    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    if remove_color:
        # Remove a specific background color (make it transparent)
        c = remove_color.lstrip('#')
        if len(c) == 6:
            target = tuple(int(c[i:i+2], 16) for i in (0, 2, 4))
        else:
            target = (255, 255, 255)  # default white
        px = img.load()
        w, h = img.size
        for y in range(h):
            for x in range(w):
                r, g, b, a = px[x, y]
                dist = ((r - target[0])**2 + (g - target[1])**2 + (b - target[2])**2) ** 0.5
                if dist <= tolerance:
                    # Fade alpha based on distance
                    new_a = int(min(a, max(0, dist / tolerance * 255)))
                    px[x, y] = (r, g, b, new_a)

    if opacity is not None:
        opacity_val = float(opacity)
        r, g, b, a = img.split()
        a = a.point(lambda p: int(p * max(0, min(1, opacity_val))))
        img = Image.merge('RGBA', (r, g, b, a))

    return image_to_response(img, 'PNG')


# --- PIPELINE EXECUTOR ---
@app.route('/api/pipeline', methods=['POST'])
def run_pipeline():
    """Execute a multi-step pipeline on an image.

    Expects JSON:
    {
        "image_base64": "...",
        "steps": [
            {"tool": "resize", "params": {"width": 800}},
            {"tool": "blur", "params": {"radius": 3}},
            {"tool": "compress", "params": {"quality": 80, "format": "WEBP"}}
        ]
    }
    """
    if not request.is_json:
        return error('JSON body required')

    data = request.json
    if 'image_base64' not in data:
        return error('image_base64 required')
    if 'steps' not in data or not data['steps']:
        return error('steps array required')

    try:
        img_data = base64.b64decode(data['image_base64'])
        img = Image.open(io.BytesIO(img_data))
    except Exception as e:
        return error(f'Invalid image: {str(e)}')

    results = []
    for i, step in enumerate(data['steps']):
        tool = step.get('tool')
        params = step.get('params', {})

        if tool not in PIPELINE_FUNCS:
            return error(f'Unknown tool: {tool} at step {i}')

        try:
            img = PIPELINE_FUNCS[tool](img, params)
            results.append({'step': i, 'tool': tool, 'status': 'ok', 'size': list(img.size)})
        except Exception as e:
            return error(f'Error at step {i} ({tool}): {str(e)}')

    fmt = 'PNG'
    quality = 90
    # Check last step for format
    if data['steps'] and data['steps'][-1].get('tool') in ('convert', 'compress'):
        fmt = data['steps'][-1].get('params', {}).get('format', 'PNG').upper()
        quality = int(data['steps'][-1].get('params', {}).get('quality', 90))

    return jsonify({
        'image_base64': image_to_base64(img, fmt, quality),
        'format': fmt,
        'size': list(img.size),
        'steps_completed': results
    })


# ============================================================
# PIPELINE FUNCTIONS (for /api/pipeline)
# ============================================================

def _resize(img, params):
    width = params.get('width')
    height = params.get('height')
    mode = params.get('mode', 'contain')
    w, h = img.size
    if width: width = int(width)
    if height: height = int(height)
    if mode == 'stretch':
        return img.resize((width or w, height or h), Image.LANCZOS)
    elif mode == 'cover':
        return ImageOps.fit(img, (width or w, height or h), Image.LANCZOS)
    else:
        if width and height:
            img.thumbnail((width, height), Image.LANCZOS)
            return img
        elif width:
            ratio = width / w
            return img.resize((width, int(h * ratio)), Image.LANCZOS)
        elif height:
            ratio = height / h
            return img.resize((int(w * ratio), height), Image.LANCZOS)
    return img

def _crop(img, params):
    x, y = int(params.get('x', 0)), int(params.get('y', 0))
    w = int(params.get('width', img.size[0]))
    h = int(params.get('height', img.size[1]))
    return img.crop((x, y, x + w, y + h))

def _rotate(img, params):
    angle = float(params.get('angle', 90))
    if params.get('flip_h'): img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if params.get('flip_v'): img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img.rotate(-angle, expand=True, resample=Image.BICUBIC)

def _blur(img, params):
    r = float(params.get('radius', 5))
    return img.filter(ImageFilter.GaussianBlur(r))

def _sharpen(img, params):
    return ImageEnhance.Sharpness(img).enhance(float(params.get('amount', 2.0)))

def _brightness(img, params):
    return ImageEnhance.Brightness(img).enhance(float(params.get('factor', 1.2)))

def _contrast(img, params):
    return ImageEnhance.Contrast(img).enhance(float(params.get('factor', 1.5)))

def _saturation(img, params):
    return ImageEnhance.Color(img).enhance(float(params.get('factor', 1.5)))

def _grayscale(img, params):
    return img.convert('L').convert('RGB')

def _sepia(img, params):
    intensity = float(params.get('intensity', 0.8))
    gray = img.convert('L')
    w, h = gray.size
    out = Image.new('RGB', (w, h))
    gp = gray.load()
    op = out.load()
    for y in range(h):
        for x in range(w):
            g = gp[x, y]
            op[x, y] = (min(255, int(g + 94*intensity)), min(255, int(g + 38*intensity)), max(0, int(g - 20*intensity)))
    return out

def _invert(img, params):
    return ImageOps.invert(img.convert('RGB'))

def _convert(img, params):
    return img  # Format handled by pipeline executor

def _compress(img, params):
    return img  # Quality handled by pipeline executor

def _pixelate(img, params):
    bs = int(params.get('block_size', 10))
    w, h = img.size
    small = img.resize((max(1, w//bs), max(1, h//bs)), Image.NEAREST)
    return small.resize((w, h), Image.NEAREST)

def _posterize(img, params):
    return ImageOps.posterize(img.convert('RGB'), int(params.get('bits', 4)))

def _threshold(img, params):
    level = int(params.get('level', 128))
    return img.convert('L').point(lambda x: 255 if x > level else 0, '1').convert('RGB')

def _edge_detect(img, params):
    return img.filter(ImageFilter.FIND_EDGES)

def _emboss(img, params):
    return img.filter(ImageFilter.EMBOSS)

def _vignette(img, params):
    strength = float(params.get('strength', 0.5))
    w, h = img.size
    mask = Image.new('L', (w, h), 255)
    draw = ImageDraw.Draw(mask)
    for i in range(min(w, h) // 2):
        alpha = int(255 * (1 - strength * (1 - i / (min(w, h) / 2)) ** 2))
        draw.ellipse([i, i, w-i, h-i], fill=max(0, min(255, alpha)))
    if img.mode != 'RGBA': img = img.convert('RGBA')
    dark = Image.new('RGBA', (w, h), (0,0,0,255))
    return Image.composite(img, dark, mask).convert('RGB')


def _hue_shift(img, params):
    degrees = float(params.get('degrees', 30))
    img = img.convert('HSV')
    px = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            hv, s, v = px[x, y]
            hv = int((hv + degrees * 255 / 360) % 256)
            px[x, y] = (hv, s, v)
    return img.convert('RGB')

def _duotone(img, params):
    dark = params.get('dark_color', '#001848').lstrip('#')
    light = params.get('light_color', '#f7a800').lstrip('#')
    d = tuple(int(dark[i:i+2], 16) for i in (0,2,4))
    l = tuple(int(light[i:i+2], 16) for i in (0,2,4))
    gray = img.convert('L')
    w, h = gray.size
    out = Image.new('RGB', (w, h))
    gp, op = gray.load(), out.load()
    for y in range(h):
        for x in range(w):
            t = gp[x,y] / 255.0
            op[x,y] = tuple(int(d[i]*(1-t)+l[i]*t) for i in range(3))
    return out

def _solarize(img, params):
    return ImageOps.solarize(img.convert('RGB'), int(params.get('threshold', 128)))

def _mirror(img, params):
    d = params.get('direction', 'horizontal')
    return img.transpose(Image.FLIP_TOP_BOTTOM if d == 'vertical' else Image.FLIP_LEFT_RIGHT)

def _sketch(img, params):
    gray = img.convert('L')
    inv = ImageOps.invert(gray)
    blr = inv.filter(ImageFilter.GaussianBlur(21))
    w, h = gray.size
    gp, bp = gray.load(), blr.load()
    out = Image.new('L', (w, h))
    op = out.load()
    for y in range(h):
        for x in range(w):
            bv = bp[x, y]
            op[x, y] = min(255, int(gp[x, y] * 256 / (256 - bv + 1))) if bv else 0
    return out.convert('RGB')

def _circle_crop(img, params):
    w, h = img.size
    size = min(w, h)
    img = ImageOps.fit(img, (size, size), Image.LANCZOS)
    mask = Image.new('L', (size, size), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, size, size], fill=255)
    if img.mode != 'RGBA': img = img.convert('RGBA')
    out = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    out.paste(img, mask=mask)
    return out

def _border(img, params):
    bw = int(params.get('width', 20))
    c = params.get('color', '#ffffff').lstrip('#')
    rgb = tuple(int(c[i:i+2], 16) for i in (0,2,4))
    return ImageOps.expand(img.convert('RGB'), border=bw, fill=rgb)

def _levels(img, params):
    bp = int(params.get('black_point', 0))
    wp = int(params.get('white_point', 255))
    gamma = float(params.get('gamma', 1.0))
    img = img.convert('RGB')
    px = img.load()
    w, h = img.size
    rng = max(1, wp - bp)
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            chs = []
            for ch in (r, g, b):
                ch = max(0, min(255, (ch - bp) * 255 // rng))
                ch = int(255 * (ch / 255.0) ** (1.0 / gamma))
                chs.append(max(0, min(255, ch)))
            px[x, y] = tuple(chs)
    return img


def _watermark(img, params):
    text = params.get('text', 'launch.pics')
    opacity = int(params.get('opacity', 128))
    position = params.get('position', 'bottom-right')
    font_size = int(params.get('font_size', 24))
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', font_size)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    w, h = img.size
    pad = 20
    positions = {'top-left': (pad, pad), 'top-right': (w-tw-pad, pad), 'bottom-left': (pad, h-th-pad), 'bottom-right': (w-tw-pad, h-th-pad), 'center': ((w-tw)//2, (h-th)//2)}
    pos = positions.get(position, positions['bottom-right'])
    draw.text(pos, text, fill=(255, 255, 255, opacity), font=font)
    if img.mode != 'RGBA': img = img.convert('RGBA')
    return Image.alpha_composite(img, overlay).convert('RGB')

def _color_adjust(img, params):
    temperature = float(params.get('temperature', 0))
    tint = float(params.get('tint', 0))
    img = img.convert('RGB')
    px = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            r = max(0, min(255, int(r + temperature * 0.5)))
            b = max(0, min(255, int(b - temperature * 0.5)))
            g = max(0, min(255, int(g + tint * 0.3)))
            px[x, y] = (r, g, b)
    return img

def _shadow(img, params):
    offset_x = int(params.get('offset_x', 10))
    offset_y = int(params.get('offset_y', 10))
    blur_radius = int(params.get('blur', 15))
    color = params.get('color', '#000000')
    w, h = img.size
    padding = max(abs(offset_x), abs(offset_y)) + blur_radius * 2
    nw, nh = w + padding * 2, h + padding * 2
    shadow_img = Image.new('RGBA', (nw, nh), (0, 0, 0, 0))
    shadow_layer = Image.new('RGBA', (w, h), color + 'AA')
    shadow_img.paste(shadow_layer, (padding + offset_x, padding + offset_y))
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(blur_radius))
    if img.mode != 'RGBA': img = img.convert('RGBA')
    shadow_img.paste(img, (padding, padding), img)
    return shadow_img

def _film_grain(img, params):
    import random
    amount = float(params.get('amount', 0.3))
    w, h = img.size
    px = img.load()
    for y in range(h):
        for x in range(w):
            noise_val = int((random.random() - 0.5) * 255 * amount)
            p = px[x, y]
            if isinstance(p, int):
                px[x, y] = max(0, min(255, p + noise_val))
            else:
                px[x, y] = tuple(max(0, min(255, c + noise_val)) for c in p[:3])
    return img

def _halftone(img, params):
    dot_size = int(params.get('dot_size', 8))
    gray = img.convert('L')
    w, h = gray.size
    out = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    gp = gray.load()
    for y in range(0, h, dot_size):
        for x in range(0, w, dot_size):
            block_sum = count = 0
            for dy in range(min(dot_size, h - y)):
                for dx in range(min(dot_size, w - x)):
                    block_sum += gp[x + dx, y + dy]; count += 1
            avg = block_sum / count if count else 128
            radius = int((1 - avg / 255) * dot_size / 2)
            if radius > 0:
                cx, cy = x + dot_size // 2, y + dot_size // 2
                draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(0, 0, 0))
    return out

def _oil_paint(img, params):
    radius = int(params.get('radius', 4))
    img = img.convert('RGB')
    w, h = img.size
    src = img.load()
    out = img.copy()
    dst = out.load()
    for y in range(radius, h - radius):
        for x in range(radius, w - radius):
            intensities = {}
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    r, g, b = src[x + dx, y + dy]
                    intensity = (r + g + b) // 3 // 32
                    if intensity not in intensities:
                        intensities[intensity] = [0, 0, 0, 0]
                    intensities[intensity][0] += r
                    intensities[intensity][1] += g
                    intensities[intensity][2] += b
                    intensities[intensity][3] += 1
            mc = max(intensities.values(), key=lambda v: v[3])
            c = mc[3]
            dst[x, y] = (mc[0] // c, mc[1] // c, mc[2] // c)
    return out

def _tilt_shift(img, params):
    blur_amount = float(params.get('blur', 10))
    focus_position = float(params.get('focus_position', 0.5))
    focus_width = float(params.get('focus_width', 0.2))
    w, h = img.size
    blurred = img.filter(ImageFilter.GaussianBlur(blur_amount))
    mask = Image.new('L', (w, h), 0)
    for y in range(h):
        t = y / h
        dist = abs(t - focus_position) / max(focus_width, 0.01)
        alpha = min(255, int(max(0, dist - 1) * 255))
        for x in range(w):
            mask.putpixel((x, y), alpha)
    img = Image.composite(blurred, img, mask)
    return ImageEnhance.Color(img).enhance(1.3)

def _color_replace(img, params):
    from_color = params.get('from_color', '#ff0000')
    to_color = params.get('to_color', '#0000ff')
    tolerance = int(params.get('tolerance', 40))
    def hex_to_rgb(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0,2,4))
    fc, tc = hex_to_rgb(from_color), hex_to_rgb(to_color)
    img = img.convert('RGB')
    px = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            dist = ((r-fc[0])**2 + (g-fc[1])**2 + (b-fc[2])**2) ** 0.5
            if dist <= tolerance:
                t = max(0, 1 - dist / tolerance)
                px[x, y] = (int(r+(tc[0]-r)*t), int(g+(tc[1]-g)*t), int(b+(tc[2]-b)*t))
    return img

def _noise(img, params):
    import random
    amount = float(params.get('amount', 0.3))
    noise_type = params.get('type', 'gaussian')
    img = img.convert('RGB')
    px = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            if noise_type == 'salt_pepper':
                if random.random() < amount * 0.1:
                    v = 255 if random.random() > 0.5 else 0
                    px[x, y] = (v, v, v)
            else:
                n = int((random.random() - 0.5) * 255 * amount)
                px[x, y] = (max(0, min(255, r+n)), max(0, min(255, g+n)), max(0, min(255, b+n)))
    return img

def _text_overlay(img, params):
    text = params.get('text', 'Hello')
    font_size = int(params.get('font_size', 48))
    color = params.get('color', '#ffffff')
    opacity_val = int(params.get('opacity', 255))
    position = params.get('position', 'center')
    x = params.get('x')
    y_pos = params.get('y')
    rotation = float(params.get('rotation', 0))
    stroke_color = params.get('stroke_color', '')
    stroke_width = int(params.get('stroke_width', 0))
    line_spacing = int(params.get('line_spacing', 4))

    def hex_to_rgba(h, a=255):
        h = h.lstrip('#')
        if len(h) == 6:
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) + (a,)
        return (255, 255, 255, a)

    fill = hex_to_rgba(color, opacity_val)
    stroke = hex_to_rgba(stroke_color, opacity_val) if stroke_color else None

    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', font_size)
    except:
        font = ImageFont.load_default()

    txt_layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(txt_layer)

    lines = text.replace('\\n', '\n').split('\n')
    line_heights = []
    line_widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])
    total_h = sum(line_heights) + line_spacing * (len(lines) - 1)
    max_w = max(line_widths) if line_widths else 0

    w, h = img.size
    pad = 20
    if x is not None and y_pos is not None:
        tx, ty = int(x), int(y_pos)
    else:
        positions = {
            'top-left': (pad, pad), 'top-center': ((w - max_w) // 2, pad),
            'top-right': (w - max_w - pad, pad),
            'center-left': (pad, (h - total_h) // 2),
            'center': ((w - max_w) // 2, (h - total_h) // 2),
            'center-right': (w - max_w - pad, (h - total_h) // 2),
            'bottom-left': (pad, h - total_h - pad),
            'bottom-center': ((w - max_w) // 2, h - total_h - pad),
            'bottom-right': (w - max_w - pad, h - total_h - pad),
        }
        tx, ty = positions.get(position, positions['center'])

    cy = ty
    for i, line in enumerate(lines):
        kwargs = {'fill': fill, 'font': font}
        if stroke and stroke_width > 0:
            kwargs['stroke_fill'] = stroke
            kwargs['stroke_width'] = stroke_width
        draw.text((tx, cy), line, **kwargs)
        cy += line_heights[i] + line_spacing

    if rotation:
        txt_layer = txt_layer.rotate(rotation, expand=False, resample=Image.BICUBIC)

    return Image.alpha_composite(img, txt_layer)

def _transparency(img, params):
    opacity = params.get('opacity')
    remove_color = params.get('remove_color', '')
    tolerance = int(params.get('tolerance', 30))

    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    if remove_color:
        c = remove_color.lstrip('#')
        if len(c) == 6:
            target = tuple(int(c[i:i+2], 16) for i in (0, 2, 4))
        else:
            target = (255, 255, 255)
        px = img.load()
        w, h = img.size
        for y in range(h):
            for x in range(w):
                r, g, b, a = px[x, y]
                dist = ((r - target[0])**2 + (g - target[1])**2 + (b - target[2])**2) ** 0.5
                if dist <= tolerance:
                    new_a = int(min(a, max(0, dist / tolerance * 255)))
                    px[x, y] = (r, g, b, new_a)

    if opacity is not None:
        opacity_f = float(opacity)
        r, g, b, a = img.split()
        a = a.point(lambda p: int(p * max(0, min(1, opacity_f))))
        img = Image.merge('RGBA', (r, g, b, a))

    return img

PIPELINE_FUNCS = {
    'resize': _resize, 'crop': _crop, 'rotate': _rotate,
    'blur': _blur, 'sharpen': _sharpen,
    'brightness': _brightness, 'contrast': _contrast, 'saturation': _saturation,
    'grayscale': _grayscale, 'sepia': _sepia, 'invert': _invert,
    'convert': _convert, 'compress': _compress,
    'watermark': _watermark, 'pixelate': _pixelate, 'posterize': _posterize,
    'threshold': _threshold, 'edge_detect': _edge_detect, 'emboss': _emboss,
    'vignette': _vignette, 'color_adjust': _color_adjust, 'shadow': _shadow,
    'film_grain': _film_grain, 'hue_shift': _hue_shift, 'duotone': _duotone,
    'halftone': _halftone, 'oil_paint': _oil_paint, 'sketch': _sketch,
    'solarize': _solarize, 'mirror': _mirror, 'tilt_shift': _tilt_shift,
    'circle_crop': _circle_crop, 'border': _border, 'levels': _levels,
    'color_replace': _color_replace, 'noise': _noise,
    'text_overlay': _text_overlay, 'transparency': _transparency,
    # Hyphen aliases (frontend/registry uses hyphens, AI generates underscores — support both)
    'edge-detect': _edge_detect, 'color-adjust': _color_adjust,
    'film-grain': _film_grain, 'hue-shift': _hue_shift,
    'oil-paint': _oil_paint, 'tilt-shift': _tilt_shift,
    'circle-crop': _circle_crop, 'color-replace': _color_replace,
    'text-overlay': _text_overlay,
}


TOOL_REGISTRY = {
    'resize': {
        'description': 'Resize image to specific dimensions',
        'params': {'width': 'int', 'height': 'int', 'mode': 'contain|cover|stretch'},
    },
    'crop': {
        'description': 'Crop image to specific region',
        'params': {'x': 'int', 'y': 'int', 'width': 'int', 'height': 'int'},
    },
    'rotate': {
        'description': 'Rotate and/or flip image',
        'params': {'angle': 'float', 'flip_h': 'bool', 'flip_v': 'bool'},
    },
    'blur': {
        'description': 'Apply gaussian or box blur',
        'params': {'radius': 'float (1-50)', 'type': 'gaussian|box'},
    },
    'sharpen': {
        'description': 'Sharpen image',
        'params': {'amount': 'float (0.5-5.0)'},
    },
    'brightness': {
        'description': 'Adjust brightness',
        'params': {'factor': 'float (0.1-3.0, 1.0=original)'},
    },
    'contrast': {
        'description': 'Adjust contrast',
        'params': {'factor': 'float (0.1-3.0, 1.0=original)'},
    },
    'saturation': {
        'description': 'Adjust color saturation',
        'params': {'factor': 'float (0.0-3.0, 1.0=original)'},
    },
    'grayscale': {
        'description': 'Convert to grayscale',
        'params': {},
    },
    'sepia': {
        'description': 'Apply sepia tone',
        'params': {'intensity': 'float (0.0-1.0)'},
    },
    'invert': {
        'description': 'Invert colors',
        'params': {},
    },
    'convert': {
        'description': 'Convert image format',
        'params': {'format': 'PNG|JPEG|WEBP|GIF', 'quality': 'int (1-100)'},
    },
    'compress': {
        'description': 'Compress image to reduce file size',
        'params': {'quality': 'int (1-100)', 'format': 'JPEG|WEBP'},
    },
    'watermark': {
        'description': 'Add text watermark',
        'params': {'text': 'string', 'position': 'top-left|top-right|center|bottom-left|bottom-right', 'opacity': 'int (0-255)', 'font_size': 'int'},
    },
    'pixelate': {
        'description': 'Pixelate image',
        'params': {'block_size': 'int (2-50)'},
    },
    'posterize': {
        'description': 'Reduce color levels',
        'params': {'bits': 'int (1-8)'},
    },
    'threshold': {
        'description': 'Convert to black and white with threshold',
        'params': {'level': 'int (0-255)'},
    },
    'edge-detect': {
        'description': 'Detect edges in image',
        'params': {},
    },
    'emboss': {
        'description': 'Apply emboss effect',
        'params': {},
    },
    'vignette': {
        'description': 'Add vignette darkening to edges',
        'params': {'strength': 'float (0.0-1.0)'},
    },
    'color-adjust': {
        'description': 'Adjust color temperature and tint',
        'params': {'temperature': 'float (-100 to 100)', 'tint': 'float (-100 to 100)'},
    },
    'shadow': {
        'description': 'Add drop shadow to image',
        'params': {'offset_x': 'int', 'offset_y': 'int', 'blur': 'int', 'color': 'hex string'},
    },
    'film-grain': {
        'description': 'Add film grain noise',
        'params': {'amount': 'float (0.0-1.0)'},
    },
    'hue-shift': {
        'description': 'Shift hue by degrees',
        'params': {'degrees': 'float (-180 to 180)'},
    },
    'duotone': {
        'description': 'Apply duotone color effect',
        'params': {'dark_color': 'hex string', 'light_color': 'hex string'},
    },
    'halftone': {
        'description': 'Apply halftone dot effect',
        'params': {'dot_size': 'int (4-20)'},
    },
    'oil-paint': {
        'description': 'Apply oil painting effect',
        'params': {'radius': 'int (2-8)'},
    },
    'sketch': {
        'description': 'Convert to pencil sketch',
        'params': {},
    },
    'solarize': {
        'description': 'Apply solarize effect',
        'params': {'threshold': 'int (0-255)'},
    },
    'mirror': {
        'description': 'Mirror image',
        'params': {'direction': 'horizontal|vertical'},
    },
    'tilt-shift': {
        'description': 'Apply tilt-shift miniature effect',
        'params': {'blur': 'float (5-30)', 'focus_position': 'float (0.0-1.0)', 'focus_width': 'float (0.1-0.5)'},
    },
    'circle-crop': {
        'description': 'Crop to circle with transparent background',
        'params': {},
    },
    'border': {
        'description': 'Add border/frame',
        'params': {'width': 'int (1-100)', 'color': 'hex string'},
    },
    'levels': {
        'description': 'Adjust input/output levels',
        'params': {'black_point': 'int (0-255)', 'white_point': 'int (0-255)', 'gamma': 'float (0.1-5.0)'},
    },
    'color-replace': {
        'description': 'Replace a specific color',
        'params': {'from_color': 'hex string', 'to_color': 'hex string', 'tolerance': 'int (1-100)'},
    },
    'noise': {
        'description': 'Add noise to image',
        'params': {'amount': 'float (0.0-1.0)', 'type': 'gaussian|salt_pepper'},
    },
    'text-overlay': {
        'description': 'Add text overlay with full styling (font, color, position, rotation, stroke)',
        'params': {'text': 'string', 'font_size': 'int (12-200)', 'color': 'hex string', 'opacity': 'int (0-255)', 'position': 'top-left|top-center|top-right|center-left|center|center-right|bottom-left|bottom-center|bottom-right', 'x': 'int (exact x position)', 'y': 'int (exact y position)', 'rotation': 'float (degrees)', 'stroke_color': 'hex string', 'stroke_width': 'int (0-10)', 'line_spacing': 'int'},
    },
    'transparency': {
        'description': 'Set image transparency / remove background color to transparent',
        'params': {'opacity': 'float (0.0-1.0)', 'remove_color': 'hex string (color to make transparent)', 'tolerance': 'int (1-100)'},
    },
}


# ============================================================
# AI WORKFLOW GENERATOR
# ============================================================

OPENROUTER_KEY = os.environ.get('OPENROUTER_API_KEY', '')

SYSTEM_PROMPT = """You are an image processing pipeline generator. Given a natural language description of how to modify an image, output a JSON array of processing steps.

Available tools and their parameters:
- resize: {width: int, height: int, mode: "contain"|"cover"|"stretch"}
- crop: {x: int, y: int, width: int, height: int}
- rotate: {angle: float, flip_h: bool, flip_v: bool}
- blur: {radius: float (1-50)}
- sharpen: {amount: float (0.5-5.0)}
- brightness: {factor: float (0.1-3.0, 1.0=original)}
- contrast: {factor: float (0.1-3.0, 1.0=original)}
- saturation: {factor: float (0.0-3.0, 1.0=original)}
- grayscale: {}
- sepia: {intensity: float (0.0-1.0)}
- invert: {}
- pixelate: {block_size: int (2-50)}
- posterize: {bits: int (1-8)}
- threshold: {level: int (0-255)}
- edge_detect: {}
- emboss: {}
- vignette: {strength: float (0.0-1.0)}
- watermark: {text: string, position: "top-left"|"top-right"|"center"|"bottom-left"|"bottom-right", opacity: int (0-255), font_size: int}
- hue_shift: {degrees: float (-180 to 180)}
- duotone: {dark_color: hex, light_color: hex}
- halftone: {dot_size: int (4-20)}
- oil_paint: {radius: int (2-8)}
- sketch: {}
- solarize: {threshold: int (0-255)}
- mirror: {direction: "horizontal"|"vertical"}
- tilt_shift: {blur: float (5-30), focus_position: float (0-1), focus_width: float (0.1-0.5)}
- circle_crop: {}
- border: {width: int (1-100), color: hex}
- levels: {black_point: int (0-255), white_point: int (0-255), gamma: float (0.1-5)}
- color_replace: {from_color: hex, to_color: hex, tolerance: int (1-100)}
- noise: {amount: float (0-1), type: "gaussian"|"salt_pepper"}
- text_overlay: {text: string, font_size: int (12-200), color: hex, opacity: int (0-255), position: "top-left"|"top-center"|"top-right"|"center-left"|"center"|"center-right"|"bottom-left"|"bottom-center"|"bottom-right", x: int, y: int, rotation: float, stroke_color: hex, stroke_width: int (0-10)}
- transparency: {opacity: float (0-1), remove_color: hex (color to make transparent), tolerance: int (1-100)}
- convert: {format: "PNG"|"JPEG"|"WEBP", quality: int (1-100)}
- compress: {quality: int (1-100), format: "JPEG"|"WEBP"}

Rules:
1. Output ONLY a valid JSON array of steps. No explanation, no markdown.
2. Each step: {"tool": "name", "params": {...}}
3. Steps execute in order. Choose logical order.
4. For "vintage" look: combine sepia + vignette + film grain adjustments
5. For "professional": resize + contrast boost + sharpen
6. Default resize to reasonable web sizes (800-1920px wide)
7. Always end with convert/compress if user mentions format or file size
8. When user says "add text", use text_overlay with appropriate styling
9. For transparency/layers, use transparency tool. Always output PNG when transparency is needed.
10. Use text_overlay for any text addition (titles, captions, labels, memes, etc.)

Example input: "Make it 800px wide, add a vintage look, and save as JPEG"
Example output: [{"tool":"resize","params":{"width":800}},{"tool":"sepia","params":{"intensity":0.6}},{"tool":"vignette","params":{"strength":0.4}},{"tool":"contrast","params":{"factor":0.9}},{"tool":"compress","params":{"quality":85,"format":"JPEG"}}]

Example input: "Resize and add text saying Hello World"
Example output: [{"tool":"resize","params":{"width":800}},{"tool":"text_overlay","params":{"text":"Hello World","font_size":64,"color":"#ffffff","position":"center","stroke_color":"#000000","stroke_width":2}}]"""


@app.route('/api/generate-workflow', methods=['POST'])
def generate_workflow():
    """Generate a pipeline from natural language description using AI."""
    if not request.is_json:
        return error('JSON body required')

    description = request.json.get('description', '')
    if not description:
        return error('description field required')

    import urllib.request

    payload = json.dumps({
        'model': 'google/gemini-2.0-flash-001',
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': description}
        ],
        'temperature': 0.3,
        'max_tokens': 1000,
    }).encode()

    req = urllib.request.Request(
        'https://openrouter.ai/api/v1/chat/completions',
        data=payload,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {OPENROUTER_KEY}',
            'HTTP-Referer': 'https://launch.pics',
            'X-Title': 'launch.pics Image Pipeline',
        }
    )

    try:
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read())
        content = result['choices'][0]['message']['content'].strip()

        # Extract JSON from response (handle markdown code blocks)
        if content.startswith('```'):
            content = content.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        if content.startswith('['):
            steps = json.loads(content)
        else:
            # Try to find JSON array in response
            import re
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                steps = json.loads(match.group())
            else:
                return error('AI did not return valid pipeline JSON')

        # Validate steps
        valid_steps = []
        for step in steps:
            if isinstance(step, dict) and 'tool' in step and step['tool'] in PIPELINE_FUNCS:
                valid_steps.append({
                    'tool': step['tool'],
                    'params': step.get('params', {})
                })

        if not valid_steps:
            return error('No valid steps generated')

        return jsonify({
            'description': description,
            'steps': valid_steps,
            'step_count': len(valid_steps),
        })

    except Exception as e:
        return error(f'AI generation failed: {str(e)}')




# ============================================================
# PUBLIC API v1: Authentication, Rate Limiting, Versioned Routes
# ============================================================

API_KEYS_FILE = "/opt/launch-pics-api-keys.json"

def load_api_keys():
    try:
        with open(API_KEYS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_api_keys(keys):
    with open(API_KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=2)

def generate_api_key():
    return "lp_" + secrets.token_hex(24)

def get_api_key_from_request():
    return request.headers.get("X-API-Key") or request.args.get("api_key")

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = get_api_key_from_request()
        if not key:
            return jsonify({"error": "API key required. Get one at https://launch.pics/docs/"}), 401
        keys = load_api_keys()
        if key not in keys:
            return jsonify({"error": "Invalid API key"}), 403
        key_data = keys[key]
        if not key_data.get("active", True):
            return jsonify({"error": "API key has been deactivated"}), 403
        key_data["requests"] = key_data.get("requests", 0) + 1
        key_data["last_used"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        keys[key] = key_data
        save_api_keys(keys)
        request.api_key_data = key_data
        return f(*args, **kwargs)
    return decorated

limiter = Limiter(
    app=app,
    key_func=lambda: get_api_key_from_request() or get_remote_address(),
    default_limits=[],
    storage_uri="memory://",
)


@app.route("/api/v1/health", methods=["GET"])
def v1_health():
    return jsonify({
        "status": "ok",
        "version": "1.0.0",
        "api": "launch.pics Image Processing API",
        "tools_count": len(TOOL_REGISTRY),
        "endpoints": {
            "docs": "https://launch.pics/docs/",
            "tools": "/api/v1/tools",
            "process": "/api/v1/{tool}",
            "pipeline": "/api/v1/pipeline",
            "generate": "/api/v1/generate-workflow",
            "batch": "/api/v1/batch",
        }
    })


@app.route("/api/v1/tools", methods=["GET"])
def v1_list_tools():
    return jsonify({"tools": TOOL_REGISTRY, "count": len(TOOL_REGISTRY)})


@app.route("/api/v1/<tool_name>", methods=["POST"])
@require_api_key
@limiter.limit("200/hour")
def v1_process(tool_name):
    normalized = tool_name.replace("-", "_")
    func = PIPELINE_FUNCS.get(normalized) or PIPELINE_FUNCS.get(tool_name)
    if not func:
        return jsonify({"error": f"Unknown tool: {tool_name}", "available": list(TOOL_REGISTRY.keys())}), 404

    img = get_image_from_request()
    if not img:
        return jsonify({"error": "No image provided. Send as multipart file (field: image) or JSON (image_base64)."}), 400

    params = get_params()
    param_dict = {k: v for k, v in (params.items() if hasattr(params, "items") else []) if k not in ("image", "image_base64", "api_key")}

    try:
        result = func(img, param_dict)
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    out_fmt = (param_dict.get("output_format") or param_dict.get("format", "PNG")).upper()
    quality = int(param_dict.get("quality", 90))

    accept = request.headers.get("Accept", "")
    if "image/" in accept and "json" not in accept:
        return image_to_response(result, out_fmt, quality)

    return jsonify({
        "image_base64": image_to_base64(result, out_fmt, quality),
        "format": out_fmt,
        "size": {"width": result.size[0], "height": result.size[1]},
        "tool": tool_name,
    })


@app.route("/api/v1/pipeline", methods=["POST"])
@require_api_key
@limiter.limit("100/hour")
def v1_pipeline():
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400

    data = request.json
    if "image_base64" not in data:
        return jsonify({"error": "image_base64 required"}), 400
    if "steps" not in data or not data["steps"]:
        return jsonify({"error": "steps array required (at least one step)"}), 400
    if len(data["steps"]) > 20:
        return jsonify({"error": "Maximum 20 steps per pipeline"}), 400

    try:
        img_data = base64.b64decode(data["image_base64"])
        img = Image.open(io.BytesIO(img_data))
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400

    original_size = list(img.size)
    results = []

    for i, step in enumerate(data["steps"]):
        tool = step.get("tool", "").replace("-", "_")
        params = step.get("params", {})
        func = PIPELINE_FUNCS.get(tool) or PIPELINE_FUNCS.get(step.get("tool", ""))
        if not func:
            return jsonify({"error": f"Unknown tool \'{step.get('tool')}\' at step {i}", "available": list(TOOL_REGISTRY.keys())}), 400

        try:
            img = func(img, params)
            results.append({"step": i, "tool": step.get("tool"), "status": "ok", "size": {"width": img.size[0], "height": img.size[1]}})
        except Exception as e:
            return jsonify({"error": f"Step {i} ({step.get('tool')}) failed: {str(e)}", "steps_completed": results}), 500

    out_fmt = data.get("output_format", "PNG").upper()
    quality = int(data.get("quality", 90))

    return jsonify({
        "image_base64": image_to_base64(img, out_fmt, quality),
        "format": out_fmt,
        "original_size": {"width": original_size[0], "height": original_size[1]},
        "final_size": {"width": img.size[0], "height": img.size[1]},
        "steps_completed": results,
    })


@app.route("/api/v1/generate-workflow", methods=["POST"])
@require_api_key
@limiter.limit("30/hour")
def v1_generate_workflow():
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400

    description = request.json.get("description", "")
    if not description:
        return jsonify({"error": "description field required"}), 400

    import urllib.request as urlreq

    payload = json.dumps({
        "model": "google/gemini-2.0-flash-001",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": description}
        ],
        "temperature": 0.3,
        "max_tokens": 1000,
    }).encode()

    req = urlreq.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "HTTP-Referer": "https://launch.pics",
            "X-Title": "launch.pics Image Pipeline API",
        }
    )

    try:
        resp = urlreq.urlopen(req, timeout=30)
        result = json.loads(resp.read())
        content = result["choices"][0]["message"]["content"].strip()

        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        if content.startswith("["):
            steps = json.loads(content)
        else:
            import re as _re
            match = _re.search(r"\[.*\]", content, _re.DOTALL)
            if match:
                steps = json.loads(match.group())
            else:
                return jsonify({"error": "AI did not return valid pipeline"}), 500

        valid_steps = []
        for step in steps:
            if isinstance(step, dict) and "tool" in step:
                tool_key = step["tool"].replace("-", "_")
                if tool_key in PIPELINE_FUNCS or step["tool"] in PIPELINE_FUNCS:
                    valid_steps.append({"tool": step["tool"], "params": step.get("params", {})})

        if not valid_steps:
            return jsonify({"error": "No valid steps generated"}), 500

        return jsonify({
            "description": description,
            "steps": valid_steps,
            "step_count": len(valid_steps),
        })

    except Exception as e:
        return jsonify({"error": f"AI generation failed: {str(e)}"}), 500


@app.route("/api/v1/batch", methods=["POST"])
@require_api_key
@limiter.limit("50/hour")
def v1_batch():
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400

    data = request.json
    images = data.get("images", [])
    if not images:
        return jsonify({"error": "images array required"}), 400
    if len(images) > 10:
        return jsonify({"error": "Maximum 10 images per batch"}), 400

    tool = data.get("tool")
    params = data.get("params", {})
    steps = data.get("steps")

    if not tool and not steps:
        return jsonify({"error": "Provide either tool+params or steps array"}), 400

    out_fmt = data.get("output_format", "PNG").upper()
    quality = int(data.get("quality", 90))
    results = []

    for idx, img_b64 in enumerate(images):
        try:
            img = Image.open(io.BytesIO(base64.b64decode(img_b64)))
        except Exception as e:
            results.append({"index": idx, "error": f"Invalid image: {str(e)}"})
            continue

        try:
            if steps:
                for step in steps:
                    st = step.get("tool", "").replace("-", "_")
                    func = PIPELINE_FUNCS.get(st) or PIPELINE_FUNCS.get(step.get("tool", ""))
                    if not func:
                        raise ValueError(f"Unknown tool: {step.get('tool')}")
                    img = func(img, step.get("params", {}))
            else:
                normalized = tool.replace("-", "_")
                func = PIPELINE_FUNCS.get(normalized) or PIPELINE_FUNCS.get(tool)
                if not func:
                    results.append({"index": idx, "error": f"Unknown tool: {tool}"})
                    continue
                img = func(img, params)

            results.append({
                "index": idx,
                "image_base64": image_to_base64(img, out_fmt, quality),
                "size": {"width": img.size[0], "height": img.size[1]},
            })
        except Exception as e:
            results.append({"index": idx, "error": str(e)})

    return jsonify({
        "results": results,
        "processed": sum(1 for r in results if "image_base64" in r),
        "errors": sum(1 for r in results if "error" in r),
        "format": out_fmt,
    })


# Admin endpoints
ADMIN_SECRET = os.environ.get("LP_ADMIN_SECRET", "")

@app.route("/api/v1/keys/create", methods=["POST"])
def create_api_key_endpoint():
    if request.headers.get("X-Admin-Secret") != ADMIN_SECRET:
        return jsonify({"error": "Unauthorized"}), 403

    data = request.get_json(silent=True) or {}
    name = data.get("name", "unnamed")
    email = data.get("email", "")

    key = generate_api_key()
    keys = load_api_keys()
    keys[key] = {
        "name": name,
        "email": email,
        "active": True,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "requests": 0,
        "rate_limit": data.get("rate_limit", "200/hour"),
    }
    save_api_keys(keys)

    return jsonify({"api_key": key, "name": name, "message": "Store this key securely."})


@app.route("/api/v1/keys/usage", methods=["GET"])
@require_api_key
def key_usage():
    data = request.api_key_data
    return jsonify({
        "name": data.get("name"),
        "requests": data.get("requests", 0),
        "rate_limit": data.get("rate_limit", "200/hour"),
        "created": data.get("created"),
        "last_used": data.get("last_used"),
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3240, debug=False)
