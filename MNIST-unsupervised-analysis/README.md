# MNIST-unsupervised-analysis

پروژه‌ی **MNIST-unsupervised-analysis** مجموعه‌ای از نوت‌بوک‌ها برای **تحلیل بدون‌ناظر (Unsupervised)** روی دیتاست MNIST است. در این پروژه:
- با **Autoencoder** ویژگی‌های نهفته (Latent Features) استخراج می‌شوند.
- فضای نهفته به‌صورت **۲ بعدی** برای نمایش مستقیم بصری‌سازی می‌شود.
- برای فضای نهفته با ابعاد بالاتر (**۶۴ بعدی**) از روش‌های **کاهش بعد (PCA و t-SNE)** استفاده می‌شود.
- همچنین یک نمونه **خوشه‌بندی با KMeans** روی نمایش ویژگی‌ها/داده انجام می‌گیرد.

---

## Features / Notebooks
- **MNIST-clustering-kmeans.ipynb**  
  خوشه‌بندی داده‌ها با الگوریتم KMeans و بررسی ساختار خوشه‌ها.

- **MNIST-Visulized-2D-Encoder.ipynb**  
  آموزش/استفاده از Autoencoder با **Encoder دو بعدی** برای نمایش مستقیم فضای نهفته (2D Latent Space).

- **MNIST-Visulized-64D-Encoder.ipynb**  
  آموزش/استفاده از Autoencoder با **Encoder شصت‌وچهار بعدی** و سپس کاهش بعد با **PCA** و **t-SNE** برای بصری‌سازی.

---

## Requirements
برای اجرای نوت‌بوک‌ها، این موارد پیشنهاد می‌شود:
- Python 3.8+
- `tensorflow`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn` (اختیاری)
- `jupyter` / `notebook` یا `jupyterlab`

نصب سریع:
```bash
pip install tensorflow numpy scikit-learn matplotlib seaborn jupyter
``` 
## ساختار پروژه
```text
MNIST-unsupervised-analysis/
├── _data/
│   ├── mnist.npz
│   └── mnist_classes.npz
├── _images/
│   ├── 0.png
│   ├── 1.png
│   ├── ...
│   ├── 9.png
│   ├── latent.png
│   ├── decoding.png
│   ├── PCA.png
│   └── t-sen.png
└── _models/
├── MNIST-clustering-kmeans.ipynb
├── MNIST-Visulized-2D-Encoder.ipynb
└── MNIST-Visulized-64D-Encoder.ipynb
```
### Samples

**نمونه بازسازی تصاویر توسط Decoder (Decoding / Reconstruction):**
![Decoding](images/decoding.png)


**نمایش فضای نهفته دو بعدی (Encoder 2D):**
![2D Latent Space](images/latent.png)

**نمایش فضای نهفته دو بعدی برای هر عدد (0 تا 9) — (Encoder 2D):**
<p align="center">
  <img src="images/0.png" alt="Digit 0" width="180"/>
  <img src="images/1.png" alt="Digit 1" width="180"/>
  <img src="images/2.png" alt="Digit 2" width="180"/>
  <img src="images/3.png" alt="Digit 3" width="180"/>
  <img src="images/4.png" alt="Digit 4" width="180"/>
</p>
<p align="center">
  <img src="images/5.png" alt="Digit 5" width="180"/>
  <img src="images/6.png" alt="Digit 6" width="180"/>
  <img src="images/7.png" alt="Digit 7" width="180"/>
  <img src="images/8.png" alt="Digit 8" width="180"/>
  <img src="images/9.png" alt="Digit 9" width="180"/>
</p>

**کاهش بعد از فضای نهفته 64 بعدی با PCA (Encoder 64D → PCA):**
![PCA](images/PCA.png)

**کاهش بعد از فضای نهفته 64 بعدی با t-SNE (Encoder 64D → t-SNE):**
![t-SNE](images/t-sen.png)

