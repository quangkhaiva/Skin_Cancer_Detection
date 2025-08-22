from __future__ import division, print_function
# coding=utf-8
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # ép dùng tf.keras 2.x
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # ẩn bớt log
import numpy as np

# Flask
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Keras
from keras import layers
from keras.models import load_model
from keras.utils import load_img, img_to_array
# Nếu lúc train dùng backbone có preprocess_input thì import đúng hàm và dùng nó.

app = Flask(__name__)

# ===== Compat cho .h5 cũ (batch_shape -> batch_input_shape) =====
class InputLayerCompat(layers.InputLayer):
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs and 'batch_input_shape' not in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super().__init__(*args, **kwargs)

# ===== Model & nhãn =====
MODEL_PATH = r'D:\Hoc tap\AI project\Skin_Cancer_Detection\Skin_Cancer_Detection_model.h5'
model = load_model(
    MODEL_PATH,
    custom_objects={'InputLayer': InputLayerCompat},
    compile=False  # chỉ infer, tránh phụ thuộc loss/optimizer cũ
)

lesion_classes_dict = {
    0: 'Melanocytic Nevi (Nốt ruồi sắc tố)',
    1: 'Melanoma (Ung thư hắc tố)',
    2: 'Benign Keratosis-like Lesions (Tổn thương dạng sừng lành tính)',
    3: 'Basal Cell Carcinoma (Ung thư tế bào đáy)',
    4: 'Actinic Keratoses (Sừng quang hóa)',
    5: 'Vascular Lesions (Tổn thương mạch máu)',
    6: 'Dermatofibroma (U sợi bì)'
}

# ===== Dự đoán =====
def model_predict(img_path, model):
    img = load_img(img_path, target_size=(224, 224))   # chỉ (h, w)
    x = img_to_array(img)
    x = x / 255.0  # nếu lúc train dùng rescale=1./255; nếu dùng preprocess_input thì thay dòng này
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds

# ===== Routes =====
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        upload_dir = os.path.join(basepath, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)
        pred_class = preds.argmax(axis=-1)[0]
        result = str(lesion_classes_dict[pred_class])
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)