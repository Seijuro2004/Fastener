# 🔩 Fastener Detector

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=black)
![No ML](https://img.shields.io/badge/Machine%20Learning-none-lightgrey)

A small web app that identifies the type of a fastener (nut, bolt, screw or washer) from a photo.
Upload an image and the app returns the predicted label, a match percentage, the detected object
outlined on the photo, and the FAST keypoints that were used.

It uses **classical computer vision only**: no neural networks, no training loop, no ML libraries.

## How it works

1. **Colour detection.** The photos have a teal backdrop. The image is converted to HSV, the backdrop hue
   is estimated from the image border and masked out, and the largest remaining blob is taken as the fastener.
2. **FAST + ORB.** The object is cropped, contrast-normalised (CLAHE), and FAST finds keypoints.
   ORB computes a binary descriptor for each one.
3. **Matching.** Descriptors are compared with the stored reference photos using OpenCV's brute-force
   Hamming matcher and Lowe's ratio test. The match score is combined with colour and shape cues
   (hue, saturation, aspect ratio, relative size, solidity).
4. **Result.** The best-scoring class (mean of its top 3 reference matches) is returned.

## Supported classes

| ID | Label | ID | Label |
|----|-------|----|-------|
| 0 | Flange Nut | 5 | Hex Bolt (Long) |
| 1 | Hex Nut | 6 | Countersunk Screw |
| 2 | Flat Washer | 7 | Phillips Washer-Head Screw |
| 3 | Spring Washer | 8 | Machine Screw |
| 4 | Hex Bolt (Short) | 9 | Coupling Nut |

The dataset only had numeric labels, so these names were assigned by inspecting the images.
Edit the `LABELS` dict in `detector.py` to rename them.

## Tech stack

| Layer | Tools |
|-------|-------|
| Frontend | HTML, CSS, vanilla JavaScript |
| Server | Flask |
| Vision | OpenCV (`opencv-python-headless`), NumPy |

## Project structure

```
fastener_detector/
├── app.py              # Flask server and /api/detect endpoint
├── detector.py         # segmentation, FAST/ORB features, matching
├── refs_cache.pkl      # prebuilt reference features (so the dataset isn't needed to run)
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── app.js
└── dataset/            # optional, git-ignored; only needed to rebuild refs_cache.pkl
```

## Run it on another computer

**Requirements:** Python 3.10 or newer and Git.

```bash
# 1. Get the code
git clone <your-repo-url>
cd fastener_detector

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell / cmd)
source .venv/bin/activate       # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
python app.py
```

Open <http://127.0.0.1:5000>, drop in a photo and press **Detect**.

The repo includes `refs_cache.pkl`, so no dataset download is needed. Make sure it is committed and
not git-ignored, otherwise the app will start with no reference data and every request will fail.

### Rebuilding the reference cache (optional)

Do this if you add new photos or change the detection settings in `detector.py`.

1. Unzip the dataset so the layout is `dataset/fasteners/0/…`, `dataset/fasteners/1/…`, and so on.
2. Delete `refs_cache.pkl`.
3. Run `python app.py`. The cache is rebuilt on startup (it takes a few seconds).

## API

`POST /api/detect` with multipart form field `image`.

```json
{
  "label": "Hex Nut",
  "confidence": 85.6,
  "keypoints": 981,
  "annotated": "data:image/jpeg;base64,...",
  "features": "data:image/jpeg;base64,..."
}
```

Errors return `{"error": "..."}` with status 400 (bad upload) or 422 (no fastener found).

## Known limitations

- Leave-one-out accuracy on the supplied dataset is about 80%.
- Flat washers (only 3 photos) and short hex bolts are the weakest classes.
- It expects one fastener on a plain, evenly lit background similar to the dataset. Other backdrops
  will degrade segmentation.
- "Match percentage" is a relative score between classes, not a calibrated probability.
