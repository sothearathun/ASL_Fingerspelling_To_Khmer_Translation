import os
import numpy as np
import cv2
import uharfbuzz as hb
import freetype

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

KHMER_FONT_PATH = "C:/Windows/Fonts/KhmerUI.ttf"
_hb_font_cache = {}
_ft_face_cache = {}

def _get_hb_font():
    if 'font' not in _hb_font_cache:
        with open(KHMER_FONT_PATH, 'rb') as f:
            font_data = f.read()
        face = hb.Face(font_data)
        font = hb.Font(face)
        font.scale = (face.upem, face.upem)
        hb.ot_font_set_funcs(font)
        _hb_font_cache['font'] = font
        _hb_font_cache['upem'] = face.upem
    return _hb_font_cache['font'], _hb_font_cache['upem']

def _get_ft_face(font_size):
    if font_size not in _ft_face_cache:
        face = freetype.Face(KHMER_FONT_PATH)
        face.set_pixel_sizes(0, font_size)
        _ft_face_cache[font_size] = face
    return _ft_face_cache[font_size]

def draw_khmer_text(frame, text, position, font_size=24, color=(255, 255, 0)):
    """Draw Khmer text onto a BGR OpenCV frame with correct complex-script shaping.

    cv2.putText only supports Hershey fonts (no non-Latin glyphs at all), and
    Pillow's ImageDraw.text on this system has no raqm/HarfBuzz support, so it
    draws Khmer codepoints in raw storage order with no vowel reordering or
    subscript-consonant stacking (garbled output). This shapes the text with
    HarfBuzz directly (forcing the Khmer script tag, since guessing from mixed
    Latin+Khmer text picks the wrong script) and rasterizes glyphs via FreeType.
    `text` should be Khmer-only; render any Latin prefix/label separately with
    cv2.putText. `color` is a BGR tuple to match the rest of the codebase.
    """
    if not text:
        return frame
    if not os.path.exists(KHMER_FONT_PATH):
        cv2.putText(frame, "[Khmer font missing]", position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        return frame

    hb_font, upem = _get_hb_font()
    ft_face = _get_ft_face(font_size)

    buf = hb.Buffer()
    buf.add_str(text)
    buf.script = 'Khmr'
    buf.direction = 'ltr'
    buf.language = 'km'
    hb.shape(hb_font, buf)

    scale = font_size / upem
    pen_x, pen_y = position
    color_arr = np.array(color, dtype=np.float32)
    frame_h, frame_w = frame.shape[:2]

    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        x_offset = pos.x_offset * scale
        y_offset = pos.y_offset * scale
        x_advance = pos.x_advance * scale
        y_advance = pos.y_advance * scale

        ft_face.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER)
        bitmap = ft_face.glyph.bitmap
        h, w = bitmap.rows, bitmap.width

        if h > 0 and w > 0:
            gx = int(pen_x + x_offset + ft_face.glyph.bitmap_left)
            gy = int(pen_y - y_offset - ft_face.glyph.bitmap_top)

            x0, y0 = max(gx, 0), max(gy, 0)
            x1, y1 = min(gx + w, frame_w), min(gy + h, frame_h)
            if x1 > x0 and y1 > y0:
                glyph_alpha = np.array(bitmap.buffer, dtype=np.uint8).reshape(h, w)
                src = glyph_alpha[y0 - gy:y1 - gy, x0 - gx:x1 - gx].astype(np.float32) / 255.0
                roi = frame[y0:y1, x0:x1].astype(np.float32)
                alpha = src[..., None]
                frame[y0:y1, x0:x1] = (roi * (1 - alpha) + color_arr * alpha).astype(np.uint8)

        pen_x += x_advance
        pen_y += y_advance

    return frame

def normalize_landmarks(landmarks):
    """Make hand landmarks invariant to on-screen position and camera distance.

    Centers all points on the wrist (landmark 0) and scales by the
    wrist-to-middle-finger-MCP (landmark 9) distance, a stable palm-bone
    reference that doesn't shift when fingers curl.
    """
    points = np.array(landmarks, dtype=np.float32).reshape(21, 3)
    wrist = points[0]
    centered = points - wrist
    scale = np.linalg.norm(centered[9])
    if scale < 1e-6:
        scale = 1e-6
    return (centered / scale).flatten().tolist()

def load_data(data_dir):
    X = []
    y = []
    class_names = os.listdir(data_dir)
    
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.npy'):
                file_path = os.path.join(class_dir, file_name)
                landmarks = np.load(file_path)
                X.append(landmarks)
                y.append(label)
    
    return np.array(X), np.array(y), class_names