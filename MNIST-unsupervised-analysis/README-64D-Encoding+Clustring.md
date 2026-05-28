# 🧠 MNIST_64D_encoding + Clustering
## 📌 معرفی پروژه
این پروژه یک پیاده‌سازی از یادگیری نمایش (Representation Learning) با استفاده از Convolutional Autoencoder بر روی دیتاست معروف MNIST است. هدف این مدل یادگیری یک نمایش فشرده از تصاویر دست‌نویس اعداد است به طوری که بتوان از این نمایش نهفته برای وظایف دیگری مانند خوشه‌بندی (Clustering) استفاده کرد.

در این پروژه ابتدا یک Autoencoder کانولوشنی برای بازسازی تصاویر آموزش داده می‌شود و سپس بردارهای نهفته (Latent Representations) استخراج شده و برای خوشه‌بندی ارقام مورد استفاده قرار می‌گیرند.

## 🧩 دیتاست مورد استفاده
در این پروژه از دیتاست MNIST استفاده شده است که شامل تصاویر دست‌نویس ارقام از 0 تا 9 می‌باشد.

### ویژگی‌های دیتاست:

تعداد نمونه‌های آموزش: `60000`

تعداد نمونه‌های تست: `10000`

اندازه تصاویر: `28 × 28`
تک کاناله (Grayscale)

### بارگذاری داده‌ها در کد:

```python
with np.load("../data/mnist.npz") as data:
    x_train_raw, y_train = data["x_train"], data["y_train"]
    x_test_raw, y_test = data["x_test"], data["y_test"]
```

## ⚙️ پیش‌پردازش داده‌ها
### قبل از ورود داده‌ها به مدل، چند مرحله پیش‌پردازش انجام می‌شود:

تغییر شکل تصاویر

تبدیل به فرمت مناسب برای شبکه کانولوشنی

نرمال‌سازی مقادیر پیکسل

```python
x_train = x_train_raw.reshape(-1, 28, 28, 1) / 255.0
x_test  = x_test_raw.reshape(-1, 28, 28, 1) / 255.0
```
در این مرحله مقدار پیکسل‌ها به بازه `[0 , 1]` تبدیل می‌شود که باعث پایداری بیشتر آموزش شبکه عصبی می‌شود.

# 🧠 الگوریتم مورد استفاده

در این پروژه از Convolutional Autoencoder برای یادگیری نمایش داده‌ها استفاده شده است.
Autoencoderچیست؟
Autoencoder یک نوع شبکه عصبی بدون نظارت (Unsupervised Neural Network) است که هدف آن 
یادگیری بازسازی داده‌های ورودی می‌باشد.

ساختار کلی Autoencoder شامل دو بخش اصلی است:

## 1️⃣ Encoder
بخش Encoder وظیفه دارد داده‌های ورودی را به یک بردار فشرده (Latent Vector) تبدیل کند.

در واقع این بخش:

```text
ویژگی های مهم تصویر را یا میگیرد → بردار های نهفته → تصویر وروردی
```
## 2️⃣ Decoder
 تلاش می‌کند با استفاده از بردار نهفته، تصویر اولیه را بازسازی کند.

```text
هدفش تولید تصاویر اولیه از روی بردار های نهفته → تصویر → بردار های نهفته
```
## 🧱 معماری مدل
معماری شبکه شامل چندین لایه کانولوشنی برای استخراج ویژگی‌ها می‌باشد.

#### Encoder
```python
input_img = layers.Input(shape=(28,28,1))

x = layers.Conv2D(32, 3, activation="relu", padding="same")(input_img)
x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)

latent = layers.Dense(64)(x)
خروجی Encoder یک بردار 64 بعدی است که نمایش فشرده‌ای از تصویر ورودی محسوب می‌شود.

Decoder
Decoder وظیفه بازسازی تصویر از بردار نهفته را دارد.

python
latent_input = layers.Input(shape=(64,))

x = layers.Dense(7*7*64, activation="relu")(latent_input)
x = layers.Reshape((7,7,64))(x)

x = layers.Conv2DTranspose(64,3,strides=2,padding="same",activation="relu")(x)
x = layers.Conv2DTranspose(32,3,strides=2,padding="same",activation="relu")(x)

output = layers.Conv2D(1,3,padding="same",activation="sigmoid")(x)
```
### در نهایت خروجی شبکه تصویری با ابعاد:

```text
28 × 28 × 1
خواهد بود.
```

## 🏋️ آموزش مدل
مدل Autoencoder با هدف بازسازی تصویر ورودی آموزش داده می‌شود.

```python
autoencoder.compile(
    optimizer="adam",
    loss="mse",
    metrics=["accuracy"]
)
```
### پارامترهای آموزش:

Optimizer: `Adam`

Loss Function: `Mean Squared Error (MSE)`

Epochs: `5`

Batch Size: `32`

### فرآیند آموزش:

``` python
autoencoder.fit(
    x_train,
    x_train,
    epochs=5,
    batch_size=32,
    validation_data=(x_test, x_test)
)
```
در اینجا ورودی و خروجی شبکه یکسان هستند زیرا هدف بازسازی تصویر است.

## 📉 تابع خطا (Loss Function)
برای آموزش Autoencoder از Mean Squared Error (MSE) استفاده شده است:

```text
MSE = (1/N) * Σ (x - x̂)²
```
که در آن:
`x` تصویر واقعی
`x̂` تصویر بازسازی شده
می‌باشد.


## 📊 استخراج ویژگی‌ها (Feature Extraction)
پس از آموزش مدل، از Encoder برای استخراج بردارهای نهفته استفاده می‌شود.

```python
features = encoder.predict(x_test)
```
این بردارهای ویژگی نمایش فشرده‌ای از تصاویر هستند و می‌توان از آن‌ها در وظایف مختلفی مانند:

کاهش بعد
خوشه‌بندی
طبقه‌بندی
استفاده کرد.

## 🔎 خوشه‌بندی داده‌ها (Clustering)
در این پروژه برای خوشه‌بندی داده‌ها از الگوریتم K-Means استفاده شده است.

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10)
y_pred = kmeans.fit_predict(features)
```
هدف این مرحله گروه‌بندی تصاویر بر اساس شباهت ویژگی‌های استخراج شده است.

## 🧮 بهبود خوشه‌بندی با Deep Clustering
برای بهبود کیفیت خوشه‌بندی از یک Clustering Layer در شبکه استفاده شده است که با استفاده از Kullback–Leibler Divergence (KLD) آموزش داده می‌شود.

```python
dec_model.compile(
    optimizer="adam",
    loss="kld"
)
```
در این روش ابتدا مراکز خوشه‌ها با KMeans مقداردهی اولیه می‌شوند و سپس شبکه به صورت تکراری برای بهبود توزیع خوشه‌ها آموزش داده می‌شود.

## 🧪 کتابخانه‌های مورد استفاده
### کتابخانه‌های اصلی این پروژه:

`python`

`numpy`

`matplotlib`

`tensorflow`

`keras`

`scikit-learn`

`scipy`

## ▶️ نحوه اجرای پروژه
### ابتدا کتابخانه‌های مورد نیاز را نصب کنید:

```bash
pip install tensorflow numpy matplotlib scikit-learn scipy
```
## 🚀 توسعه‌های آینده
### ایده‌هایی برای گسترش پروژه:

استفاده از `Variational Autoencoder (VAE)`

استفاده از `t-SNE` یا `UMAP` برای visualization

افزایش عمق شبکه

آموزش مدل با epoch بیشتر

استفاده از `Deep Embedded Clustering (DEC)`

## 👨‍💻 هدف پروژه
### هدف این پروژه آشنایی عملی با مفاهیم زیر است:

Representation Learning

Convolutional 

Dimensionality Reduction

Deep Clustering

Feature Extraction with Neural Networks
