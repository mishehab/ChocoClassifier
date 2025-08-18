<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>🍫 ChocoClassifier – Real-Time Chocolate Recognition (Streamlit + TensorFlow)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root { --fg:#0f172a; --muted:#475569; --accent:#7c3aed; --bg:#ffffff; }
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial; color:var(--fg); background:var(--bg); line-height:1.6; margin:0; padding:2.5rem 1.25rem; }
    main { max-width:980px; margin:0 auto; }
    h1,h2,h3 { line-height:1.25; }
    h1 { font-size:2rem; margin-bottom:.25rem; }
    h2 { margin-top:2rem; }
    p.lead { color:var(--muted); margin-top:0; }
    code,kbd,pre { font-family: ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace; }
    pre { background:#0b1021; color:#e5e7eb; padding:1rem; border-radius:10px; overflow:auto; }
    code.inline { background:#f1f5f9; padding:.15rem .35rem; border-radius:.35rem; }
    a { color:var(--accent); text-decoration:none; }
    a:hover { text-decoration:underline; }
    .badges img { margin-right:6px; vertical-align:middle; }
    .kbd { border:1px solid #cbd5e1; border-bottom-width:2px; padding:.1rem .35rem; border-radius:.35rem; background:#f8fafc; }
    .pill { display:inline-block; padding:.2rem .6rem; border-radius:999px; background:#f3e8ff; color:#581c87; font-weight:600; font-size:.85rem; }
    .grid { display:grid; gap:1rem; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); }
    .card { border:1px solid #e2e8f0; border-radius:12px; padding:1rem; background:#fff; }
    table { border-collapse:collapse; width:100%; }
    th,td { border:1px solid #e2e8f0; padding:.5rem .6rem; text-align:left; }
    th { background:#f8fafc; }
    footer { margin-top:3rem; color:#64748b; font-size:.9rem; }
  </style>
</head>
<body>
<main>

  <header>
    <h1>🍫 ChocoClassifier</h1>
    <p class="lead">Real-time chocolate bar classification from live camera input using <strong>TensorFlow (MobileNetV2)</strong> and a mobile-friendly <strong>Streamlit</strong> app. Returns chocolate <em>name</em>, <em>price (BDT)</em>, <em>manufacturer</em>, and <em>calories</em>.</p>
    <p class="badges">
      <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white" />
      <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white" />
      <img alt="Python" src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />
    </p>
    <p>
      <strong>Live Link:</strong> <a href="https://chococlassifier.streamlit.app" target="_blank" rel="noopener">chococlassifier.streamlit.app</a><br/>
      <strong>Repository:</strong> <a href="https://github.com/mishehab/ChocoClassifier" target="_blank" rel="noopener">github.com/mishehab/ChocoClassifier</a>
    </p>
  </header>

  <section id="overview">
    <h2>Overview</h2>
    <p>
      This project trains a transfer-learning classifier (MobileNetV2) on a custom chocolate dataset
      and deploys it as a Streamlit app with <span class="pill">📷 st.camera_input</span> for phone/webcam capture.
      Predictions are mapped to product metadata from a CSV, returning price (BDT), manufacturer, and calories.
    </p>
  </section>

  <section id="features">
    <h2>Features</h2>
    <ul>
      <li>Live camera capture on mobile/desktop via Streamlit.</li>
      <li>TensorFlow/Keras model (<code class="inline">.keras</code>) optimized at 224×224 input.</li>
      <li>Top-1 and optional Top-3 predictions with confidence.</li>
      <li>Metadata lookup from CSV for price (BDT), manufacturer, calories.</li>
      <li>Deployable to Streamlit Cloud or runnable locally.</li>
    </ul>
  </section>

  <section id="project-structure">
    <h2>Project Structure</h2>
    <pre><code>.
├─ app.py
├─ chocolate_classifier.keras           # trained MobileNetV2
├─ label.csv                            # per-image metadata
├─ requirements.txt
├─ runtime.txt                          # (optional) pin Python version for Streamlit Cloud
└─ dataset_augmented/                   # (optional) local copy of augmented data for reference
</code></pre>
  </section>

  <section id="dataset">
    <h2>Dataset & Metadata</h2>
    <p>Each class has ~200 images (after augmentation), resized to 224×224. During inference, the app uses a per-image CSV:</p>
    <table>
      <thead>
        <tr><th>Column</th><th>Type</th><th>Description</th></tr>
      </thead>
      <tbody>
        <tr><td><code>filename</code></td><td>string</td><td>Image file name (or relative path); not required at inference but kept for completeness.</td></tr>
        <tr><td><code>label</code></td><td>string</td><td>Class name (e.g., DairyMilk, GreenBerryBar, ...).</td></tr>
        <tr><td><code>price</code></td><td>number</td><td>Price in BDT.</td></tr>
        <tr><td><code>manufacturer</code></td><td>string</td><td>Brand/producer.</td></tr>
        <tr><td><code>calories</code></td><td>number</td><td>Calorie estimate per bar.</td></tr>
      </tbody>
    </table>
    <p><strong>Note:</strong> The app derives a deterministic label order alphabetically from <code>label.csv</code>. Ensure this matches the label order used during training, or export an explicit index→label map during training and adapt the loader.</p>
  </section>

  <section id="training">
    <h2>Training (Google Colab)</h2>
    <ol>
      <li>Upload your augmented dataset (class folders, images at 224×224).</li>
      <li>Load MobileNetV2 with <code class="inline">include_top=False</code>, add GAP + Dense head, train frozen, then fine-tune top layers.</li>
      <li>Save model as <code class="inline">chocolate_classifier.keras</code> and export your label map if needed.</li>
    </ol>
    <pre><code># Example (snippet)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

base = MobileNetV2(include_top=False, input_shape=(224,224,3), weights="imagenet")
base.trainable = False
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.25)(x)
x = Dense(128, activation="relu")(x)
out = Dense(num_classes, activation="softmax")(x)
model = Model(base.input, out)
# ... warmup train, unfreeze top-N layers, fine-tune, save .keras
</code></pre>
  </section>

  <section id="app">
    <h2>Streamlit App</h2>
    <p><code>app.py</code> loads the <code>.keras</code> model and <code>label.csv</code>, captures a photo, preprocesses to 224×224, runs prediction, and displays metadata.</p>
    <pre><code>streamlit run app.py
</code></pre>
    <p>Open the provided local URL on your phone (same Wi-Fi) or deploy to Streamlit Cloud.</p>
  </section>

  <section id="install">
    <h2>Local Setup</h2>
    <div class="grid">
      <div class="card">
        <h3>Requirements</h3>
        <pre><code># requirements.txt
streamlit
tensorflow==2.20.0
pillow
numpy
pandas
</code></pre>
      </div>
      <div class="card">
        <h3>Virtual Env (Windows PowerShell)</h3>
        <pre><code>python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
</code></pre>
      </div>
      <div class="card">
        <h3>Virtual Env (macOS/Linux)</h3>
        <pre><code>python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
</code></pre>
      </div>
    </div>
  </section>

  <section id="deploy">
    <h2>Deploy to Streamlit Cloud</h2>
    <ol>
      <li>Push <code>app.py</code>, <code>chocolate_classifier.keras</code>, <code>label.csv</code>, <code>requirements.txt</code> to a public GitHub repo.</li>
      <li>(Option A) Add <code class="inline">runtime.txt</code> with <code class="inline">3.11</code> to force Python 3.11, or</li>
      <li>(Option B) Use <code class="inline">tensorflow==2.20.0</code> which supports newer Python versions on Streamlit Cloud.</li>
      <li>Create a new app at <a href="https://share.streamlit.io" target="_blank" rel="noopener">share.streamlit.io</a>, select the repo, and set main file to <code>app.py</code>.</li>
    </ol>
    <p><strong>Large model?</strong> If your <code>.keras</code> exceeds GitHub’s size limit, host it via Git LFS or a direct download (e.g., Hugging Face) at app startup.</p>
  </section>

  <section id="troubleshooting">
    <h2>Troubleshooting</h2>
    <ul>
      <li><strong>CSV not found</strong>: Ensure <code>label.csv</code> is in the repo root alongside <code>app.py</code> and committed.</li>
      <li><strong>TF wheel error on Streamlit Cloud</strong>: Pin Python to 3.11 via <code>runtime.txt</code> <em>or</em> upgrade to <code>tensorflow==2.20.0</code>.</li>
      <li><strong>Camera blocked</strong>: Allow camera permissions in your browser; on iOS/Safari, ensure HTTPS and permissions are granted.</li>
      <li><strong>Low confidence</strong>: Improve lighting, reduce glare, bring the bar closer, frame centrally. Consider adding more diverse data.</li>
    </ul>
  </section>

  <section id="license">
    <h2>License</h2>
    <p>MIT License. You are free to use, modify, and distribute with attribution.</p>
  </section>

  <footer>
    <p><strong>Live:</strong> <a href="https://chococlassifier.streamlit.app" target="_blank" rel="noopener">chococlassifier.streamlit.app</a> &nbsp;|&nbsp; <strong>Repo:</strong> <a href="https://github.com/mishehab/ChocoClassifier" target="_blank" rel="noopener">github.com/mishehab/ChocoClassifier</a></p>
  </footer>

</main>
</body>
</html>
