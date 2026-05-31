import torch
from torch.utils.data import Dataset

class MNISTPtDataset(Dataset):
    def __init__(
        self,
        data_path,
        normalize_range=(-1.0, 1.0),
        image_key=None,
        label_key=None,
        add_channel_dim=True,
        channel_last_if_4d=True,
        dtype=torch.float32,
        label_dtype=torch.long,
        flatten=False,
        transform=None,
        target_transform=None,
        map_location="cpu",
    ):
        """
        Args:
            data_path (str): مسیر فایل .pt
            normalize_range (tuple): بازه نرمال‌سازی مثل (-1,1) یا (0,1)
            image_key (str|None): اگر فایل dict بود، کلید تصاویر
            label_key (str|None): اگر فایل dict بود، کلید لیبل‌ها
            add_channel_dim (bool): اگر تصاویر [N,H,W] بودند، بعد channel اضافه شود
            channel_last_if_4d (bool): اگر تصاویر [N,H,W,C] بودند، به [N,C,H,W] تبدیل شوند
            dtype: نوع داده تصاویر
            label_dtype: نوع داده لیبل‌ها
            flatten (bool): اگر True باشد، تصویر به بردار تخت تبدیل می‌شود
            transform: transform اختیاری روی تصویر
            target_transform: transform اختیاری روی label
            map_location (str): برای torch.load
        """
        self.data_path = data_path
        self.normalize_range = normalize_range
        self.transform = transform
        self.target_transform = target_transform
        self.flatten = flatten

        obj = torch.load(data_path, map_location=map_location)

        images, labels = self._extract_images_labels(
            obj=obj,
            image_key=image_key,
            label_key=label_key
        )

        images = torch.as_tensor(images)
        labels = torch.as_tensor(labels, dtype=label_dtype)

        # شکل‌دهی تصویر
        if images.ndim == 3 and add_channel_dim:
            # [N,H,W] -> [N,1,H,W]
            images = images.unsqueeze(1)

        elif images.ndim == 4 and channel_last_if_4d and images.shape[-1] in [1, 3]:
            # [N,H,W,C] -> [N,C,H,W]
            images = images.permute(0, 3, 1, 2)

        images = images.to(dtype)

        # نرمال‌سازی
        images = self._normalize(images, normalize_range)

        # فلت کردن در صورت نیاز
        if flatten:
            images = images.view(images.size(0), -1)

        self.images = images
        self.labels = labels

    def _extract_images_labels(self, obj, image_key=None, label_key=None):
        # حالت tuple/list
        if isinstance(obj, (tuple, list)) and len(obj) >= 2:
            return obj[0], obj[1]

        # حالت dict
        if isinstance(obj, dict):
            possible_image_keys = [image_key] if image_key else []
            possible_label_keys = [label_key] if label_key else []

            possible_image_keys += ["data", "images", "x", "X"]
            possible_label_keys += ["targets", "labels", "y", "Y"]

            images = None
            labels = None

            for k in possible_image_keys:
                if k is not None and k in obj:
                    images = obj[k]
                    break

            for k in possible_label_keys:
                if k is not None and k in obj:
                    labels = obj[k]
                    break

            if images is None or labels is None:
                raise ValueError(
                    f"Could not find image/label keys in dict. "
                    f"Available keys: {list(obj.keys())}"
                )

            return images, labels

        raise ValueError(
            "Unsupported .pt structure. Expected tuple/list مثل (images, labels) "
            "یا dict شامل data/images و targets/labels."
        )

    def _normalize(self, images, normalize_range):
        """
        تصاویر را با دقت به بازه [min_val, max_val] منتقل می‌کند.
        """
        min_target, max_target = normalize_range

        if min_target >= max_target:
            raise ValueError("normalize_range is invalid. Must be like (min, max) with min < max.")

        # پیدا کردن کمینه و بیشینه واقعی داده‌های فعلی
        img_min = images.min()
        img_max = images.max()

        # جلوگیری از تقسیم بر صفر اگر تمام پیکسل‌ها یکسان باشند
        if img_max == img_min:
            return images.fill_(min_target)

        # مرحله اول: انتقال به بازه دقیق [0, 1]
        # (images - img_min) / (img_max - img_min)
        images = (images - img_min) / (img_max - img_min)

        # مرحله دوم: انتقال از [0, 1] به [min_target, max_target]
        images = images * (max_target - min_target) + min_target
        
        return images


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label
