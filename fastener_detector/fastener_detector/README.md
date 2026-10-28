# Fastener Detector

Flask + OpenCV (HSV colour segmentation, FAST keypoints, ORB descriptors). No machine learning.

```
python -m venv venv
venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python app.py                # http://127.0.0.1:5000
```

`refs_cache.pkl` holds the reference features built from the dataset. To rebuild it, delete the file,
unzip the dataset into `dataset/` (so you have `dataset/fasteners/0..9/`), and restart.
