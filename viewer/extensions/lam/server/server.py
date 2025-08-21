from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from lam import get_position_image, load_img, cal_lam
from model_loader import get_model_from_path, get_root_path, get_model
import traceback

app = Flask(__name__)
CORS(app)

root_path = get_root_path()
@app.route('/lam', methods=['POST'])
def handle_lam():
    data = request.get_json()

    type = data.get('type')
    file = data.get('file')
    path = data.get('path')
    x = data.get('x')
    y = data.get('y')
    w = data.get('w')
    h = data.get('h')
    print(type, file, path, x, y, w, h)
    img_lr, img_hr, cv2_lr, cv2_hr, tensor_lr = load_img(f'{root_path}/{file}')
    if type == 'get_position_image':
        image = get_position_image(img_hr, w, y, x)
        return send_file(image, mimetype='image/png')
    elif type == 'lam':
        try:
            if path.endswith('.yml') or path.endswith('.yaml'):
                model = get_model(path)
            else:
                model = get_model_from_path(path)
            model = model.get_bare_model(model.net_g)
            zip_file = cal_lam(model, tensor_lr, img_lr, img_hr, y, x, w, data_range=2)
            return send_file(zip_file, mimetype='application/zip')
        except Exception as e:
            stack_trace = traceback.format_exc()
            return jsonify({'error': stack_trace}), 500

if __name__ == '__main__':
    app.run(debug=True)