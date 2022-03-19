import glob
import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from src import yolo
import numpy as np


UPLOAD_FOLDER = "static/cache"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def remove_glob(pathname, recursive=True):
    for p in glob.glob(pathname, recursive=recursive):
        if os.path.isfile(p):
            os.remove(p)


# 学習済みモデルをロード
model = yolo.YOLO('model/best.pt')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        remove_glob(UPLOAD_FOLDER + "/*")
        # 出力
        outputs = {}
        diff = 0
        if 'files[]' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        files = request.files.getlist('files[]')
        for file in files:
            if file.filename == '':
                flash('ファイルがありません')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(UPLOAD_FOLDER, filename))
                file_path = os.path.join(UPLOAD_FOLDER, filename)

                # 変換したデータをモデルに渡して予測する
                pred_img, scratch_cnt = model.get_predicted_results(file_path, log=True)
                pred_answer = str(scratch_cnt)
                outputs |= {pred_img: pred_answer}

        diff = int(list(outputs.values())[0]) - int(list(outputs.values())[1])
        return render_template(
            "index.html", predicted=True, outputs=outputs, diff=np.abs(diff)
        )

    return render_template("index.html", predicted=False)


if __name__ == "__main__":
    port = 8890
    app.run(host='0.0.0.0', port=port)
