import base64, os
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from detector import FastenerDetector

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
detector = FastenerDetector(os.path.join(BASE, "dataset"), os.path.join(BASE, "refs_cache.pkl"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/detect", methods=["POST"])
def detect():
    f = request.files.get("image")
    if not f:
        return jsonify(error="No image uploaded."), 400
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="Could not read that file as an image."), 400
    r = detector.classify(img)
    if r is None:
        return jsonify(error="No fastener found. Try a plain background."), 422
    out = r["img"].copy()
    x, y, w, h = r["box"]
    cv2.drawContours(out, [r["contour"]], -1, (60, 200, 90), 2)
    cv2.rectangle(out, (x, y), (x + w, y + h), (40, 40, 40), 2)
    cv2.putText(out, r["label"], (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    ok2, cbuf = cv2.imencode(".jpg", cv2.drawKeypoints(r["crop"], r["kps"], None, (0, 255, 255)))
    b64 = lambda b: "data:image/jpeg;base64," + base64.b64encode(b).decode()
    return jsonify(label=r["label"], confidence=round(r["confidence"] * 100, 1),
                   keypoints=len(r["kps"]), annotated=b64(buf), features=b64(cbuf))


if __name__ == "__main__":
    app.run(debug=True)
