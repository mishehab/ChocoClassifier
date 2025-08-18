<h1>🍫 ChocoClassifier — Real-Time Chocolate Recognition (Streamlit + TensorFlow)</h1>

<p><strong>Live:</strong> <a href="https://chococlassifier.streamlit.app" target="_blank" rel="noopener">https://chococlassifier.streamlit.app</a><br>
<strong>Repo:</strong> <a href="https://github.com/mishehab/ChocoClassifier" target="_blank" rel="noopener">https://github.com/mishehab/ChocoClassifier</a></p>

<p>
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white">
</p>

<p>Real-time chocolate bar classification from live camera input using <strong>MobileNetV2</strong> (TensorFlow/Keras) and a mobile-friendly <strong>Streamlit</strong> UI. The app predicts chocolate <em>name</em>, <em>price (BDT)</em>, <em>manufacturer</em>, and <em>calories</em>.</p>

<hr>

<h2>✨ Features</h2>
<ul>
  <li>📷 Live camera capture on phone/desktop via <code>st.camera_input</code></li>
  <li>🧠 MobileNetV2 classifier at 224×224 input (transfer learning + fine-tuning)</li>
  <li>📊 Top-1 (and optional Top-3) predictions with confidence</li>
  <li>🗂️ Metadata lookup from <code>label.csv</code> (price, manufacturer, calories)</li>
  <li>☁️ One-click deployment to Streamlit Cloud</li>
</ul>

<h2>📦 Project Structure</h2>
<pre><code>.
├─ app.py                        <!-- Streamlit app -->
├─ chocolate_classifier.keras    <!-- Trained MobileNetV2 model -->
├─ label.csv                     <!-- filename,label,price,manufacturer,calories -->
├─ requirements.txt              <!-- dependencies -->
└─ runtime.txt                   <!-- optional: pin Python version (e.g., 3.11) -->
</code></pre>

<h2>🧰 Local Setup</h2>
<pre><code># create and activate venv
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

# install deps
pip install -r requirements.txt

# run
streamlit run app.py
</code></pre>

<h3>requirements.txt</h3>
<pre><code>streamlit
tensorflow==2.20.0
pillow
numpy
pandas
</code></pre>



<h2>🚀 Training Outline (Colab)</h2>
<ol>
  <li>Upload augmented dataset (6 classes × 200 images each, resized to 224×224).</li>
  <li>Load MobileNetV2 with <code>include_top=False</code>; add GAP → Dense head.</li>
  <li>Train with frozen base, then fine-tune top layers; use EarlyStopping and ModelCheckpoint.</li>
  <li>Save model as <code>chocolate_classifier.keras</code>.</li>
  <li>Export per-image metadata to <code>label.csv</code>.</li>
</ol>

<h2>🧪 How the App Works</h2>
<ol>
  <li>Capture or upload an image.</li>
  <li>Preprocess to 224×224 and scale to [0,1].</li>
  <li>Run inference; get class probabilities.</li>
  <li>Map predicted label → metadata from <code>label.csv</code>.</li>
  <li>Display prediction, confidence, price (BDT), manufacturer, calories.</li>
</ol>

<h2>🛠 Troubleshooting</h2>
<ul>
  <li><strong>CSV not found:</strong> Ensure <code>label.csv</code> is in the repo root (next to <code>app.py</code>).</li>
  <li><strong>TensorFlow wheel error on Streamlit Cloud:</strong> Use <code>tensorflow==2.20.0</code> or add <code>runtime.txt</code> with <code>3.11</code>.</li>
  <li><strong>Camera blocked:</strong> Allow camera permissions in the browser; Safari/iOS requires HTTPS.</li>
  <li><strong>Low confidence:</strong> Improve lighting, reduce glare, center the bar; expand dataset for better generalization.</li>
</ul>


