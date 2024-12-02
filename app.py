from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import os
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import subprocess

app = Flask(__name__)

# Cache to store the loaded slide and processing status
slide_cache = {}
process_cache = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select', methods=['POST'])
def select_file():
    if request.method == 'POST':
        filepath = request.form.get('filepath')
        if os.path.exists(filepath):
            global slide_cache, process_cache
            slide_cache['slide'] = open_slide(filepath)
            slide_cache['filename'] = os.path.basename(filepath)
            # Start processing in the background
            process = subprocess.Popen(['python3', 'process_tiles.py', filepath])
            process_cache['process'] = process
            process_cache['original_wsi'] = filepath
            process_cache['processed_wsi'] = f"processed_slides/{os.path.basename(filepath).split('.')[0]}_processed.tif"
            process_cache['processed_status'] = f"processed_slides/{os.path.basename(filepath).split('.')[0]}.txt"

            return redirect(url_for('view_image', filename=slide_cache['filename']))
    return 'Invalid file path'

@app.route('/view/<filename>')
def view_image(filename):
    slide = slide_cache.get('slide')
    processed_wsi = process_cache.get('processed_wsi')

    if os.path.exists(processed_wsi):
        slide = open_slide(processed_wsi)
        slide_cache['slide'] = slide
        slide_cache['filename'] = os.path.basename(processed_wsi)

    if slide is None:
        return 'No slide loaded', 404

    width, height = slide.dimensions
    return render_template('view.html', filename=slide_cache['filename'], width=width, height=height)

@app.route('/tile/<int:level>/<int:x>/<int:y>.jpeg')
def get_tile(level, x, y):
    slide = slide_cache.get('slide')
    if slide is None:
        return 'No slide loaded', 404

    cache_dir = os.path.join('static', 'tile_cache', slide_cache['filename'], str(level))
    os.makedirs(cache_dir, exist_ok=True)
    tile_path = os.path.join(cache_dir, f"{x}_{y}.jpeg")

    if os.path.exists(tile_path):
        return send_file(tile_path, mimetype='image/jpeg')

    dz = DeepZoomGenerator(slide, tile_size=256, overlap=0, limit_bounds=False)
    if level < dz.level_count:
        try:
            tile = dz.get_tile(level, (x, y))
            tile.save(tile_path, 'JPEG')
            return send_file(tile_path, mimetype='image/jpeg')
        except ValueError:
            return 'Tile not available', 404
    else:
        return 'Level not available', 404

@app.route('/status')
def check_status():
    global process_cache
    processed_status = process_cache.get('processed_status')

    if os.path.exists(processed_status):
        return jsonify({'status': 'completed', 'processed_wsi': process_cache.get('processed_wsi'), 'stop_polling': True})
    else:
        return jsonify({'status': 'processing', 'stop_polling': False})


if __name__ == '__main__':
    app.run(debug=True)
