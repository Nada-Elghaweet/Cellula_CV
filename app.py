import io

import numpy as np
from flask import Flask, jsonify, render_template, request, Response

import inference

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


def error(message, status=400):
    return jsonify({"error": message}), status


@app.errorhandler(413)
def too_large(_):
    return error("file is too large (limit 20 MB)", 413)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return error("no file uploaded, send it as multipart field 'file'")
    f = request.files["file"]
    if not f.filename or not f.filename.lower().endswith(".npy"):
        return error("only .npy files are supported")

    try:
        arr = np.load(io.BytesIO(f.read()), allow_pickle=False)
    except Exception:
        return error("could not read the file as a numpy array")

    try:
        threshold = float(request.args.get("threshold", inference.THRESHOLD))
    except ValueError:
        return error("threshold must be a number")
    if not 0.0 < threshold < 1.0:
        return error("threshold must be between 0 and 1")

    try:
        out = inference.run(arr, threshold)
    except ValueError as e:
        return error(str(e))

    mask_png = inference.mask_to_png_bytes(out["mask"])

    if request.args.get("format", "png") == "json":
        preview_png = inference.rgb_to_png_bytes(inference.rgb_preview(arr))
        return jsonify({
            "water_pct": out["water_pct"],
            "height": out["height"],
            "width": out["width"],
            "threshold": threshold,
            "mask_png_b64": inference.to_b64(mask_png),
            "preview_png_b64": inference.to_b64(preview_png),
        })

    resp = Response(mask_png, mimetype="image/png")
    resp.headers["X-Water-Pct"] = str(out["water_pct"])
    return resp


if __name__ == "__main__":
    inference.get_model()
    app.run(host="127.0.0.1", port=5000, debug=False)