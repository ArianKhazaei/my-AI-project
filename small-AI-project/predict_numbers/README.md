# پیش‌بینی اعداد (MNIST)

پروژه تشخیص دستنویس اعداد با شبکه عصبی روی دیتاست MNIST.

## ساختار پروژه

```
predict_numbers/
├── models/
│   └── mnist_model.h5        # فایل مدل (در .gitignore)
├── data/
│   └── mnist.npz             # دیتاست (در .gitignore)
├── notebooks/
│   └── predict_numbers.ipynb # نوت‌بوک اصلی
├── README.md
└── .gitignore
```

## پیش‌نیازها

- Python 3.8+
- TensorFlow
- NumPy
- Matplotlib

```bash
pip install tensorflow numpy matplotlib
```

## دیتاست

فایل `mnist.npz` را در پوشه `data/` قرار دهید. می‌توانید آن را از Keras دانلود کنید:

```python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
np.savez("data/mnist.npz", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
```

یا از [لینک رسمی MNIST](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz) دانلود کنید.

## نحوه اجرا

1. دیتاست را در `data/mnist.npz` قرار دهید.
2. نوت‌بوک `notebooks/predict_numbers.ipynb` را باز کنید.
3. سلول‌ها را به ترتیب اجرا کنید.

```python
sample_image = x_train_raw[np.random.randint(0, 60000)]
predict_images(model, sample_image)
```
![مثال](predict.png)

**برای استفاده از مدل آماده:** فایل مدل را در `models/mnist_model.h5` قرار دهید و از سلول‌های مربوط به بارگذاری مدل و پیش‌بینی استفاده کنید.

**برای آموزش مدل از صفر:** سلول‌های مربوط به ساخت، آموزش و ذخیره مدل را اجرا کنید.


