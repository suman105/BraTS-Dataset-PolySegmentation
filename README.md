# 🧠 Brain Tumor Segmentation Using DUCKNet on BraTS2020 Dataset

## 📌 Project Overview
This project implements brain tumor segmentation using the **DUCKNet** architecture on the **BraTS2020 dataset**. The goal is to accurately segment brain tumors from MRI images, aiding in diagnosis and treatment planning.

---

## 📖 Table of Contents
1. 🚀 [Introduction](#-introduction)
2. 📊 [Dataset](#-dataset)
3. ⚙️ [Data Preprocessing](#-data-preprocessing)
4. 🏗 [DUCKNet Architecture](#-ducknet-architecture)
5. 🎯 [Training](#-training)
6. 📈 [Results and Comparison](#-results-and-comparison)
7. 🏁 [Conclusion](#-conclusion)
8. 🛠 [Usage](#-usage)
9. 📚 [References](#-references)
10. 🤝 [Acknowledgments](#-acknowledgments)
11. 👤 [Author](#-author)
12. 📜 [License](#-license)

---

## 🚀 Introduction
Brain tumors pose significant challenges in medical diagnosis. Automated segmentation of tumors from MRI scans helps professionals by providing **consistent and accurate tumor boundaries** for diagnosis, treatment planning, and monitoring.

---

## 📊 Dataset
### **BraTS2020 Training Data**
- **Description:** MRI scans with expert tumor region annotations.
- **Data Size:** ~7 GB.

### **Download Instructions**
```bash
# Setup Kaggle API Token\!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download and Extract Dataset
!kaggle datasets download -d awsaf49/brats2020-training-data
!unzip -qq brats2020-training-data.zip -d brats2020
```

---

## ⚙️ Data Preprocessing
### **1️⃣ Install Dependencies**
```python
!pip install nibabel
```

### **2️⃣ Load the Data**
```python
from google.colab import files
files.upload()
```

### **3️⃣ Preprocessing Steps**
✅ Normalize and resize images & masks  
✅ Convert to NumPy arrays  
✅ Split into training and validation sets

---

## 🏗 DUCKNet Architecture
The **DUCKNet** model follows an encoder-decoder structure with skip connections to retain spatial information.

🛠 **Key Components:**
- **Encoder:** Reduces spatial dimensions while increasing feature maps.
- **Decoder:** Upsamples feature maps using transposed convolutions.
- **Skip Connections:** Helps retain spatial context.

![DUCKNet](https://github.com/user-attachments/assets/da7c8faa-e2be-4d02-955a-9bec35e25271)

---

## 🎯 Training
### **1️⃣ Data Augmentation**
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_gen_args = dict(horizontal_flip=True, vertical_flip=True, rotation_range=90)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
```

### **2️⃣ Training Generator**
```python
def train_generator(X_train_data, y_train_data, batch_size):
    seed_value = 42
    image_generator = image_datagen.flow(X_train_data, batch_size=batch_size, seed=seed_value)
    mask_generator = mask_datagen.flow(y_train_data, batch_size=batch_size, seed=seed_value)
    return zip(image_generator, mask_generator)

train_gen = train_generator(X_train, y_train, batch_size=8)
```

### **3️⃣ Compile & Train Model**
```python
model = DUCKNet(input_size=(256, 256, 1))
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_gen, steps_per_epoch=len(X_train) // 8, epochs=50, validation_data=(X_val, y_val))
```

---

## 📈 Results and Comparison
### **Performance Metrics**
![Results](https://github.com/user-attachments/assets/c71153d7-2a24-497c-bf73-d3059342163a)

| Metric                     | **DUCKNet** | **U-Net** | **3D MRI Segmentation** |
|----------------------------|------------|-----------|-------------------------|
| **Accuracy (Train)**       | 99.33%     | 99.31%    | 99.02%                  |
| **Accuracy (Validation)**  | 99.56%     | 99.31%    | 98.91%                  |
| **Dice Coefficient (Val)** | 88.72%     | 64.8%     | 47.03%                  |
| **Mean IoU (Val)**         | 84.30%     | 84.26%    | 78.25%                  |
| **Precision**              | 91.28%     | 99.35%    | 99.33%                  |
| **Validation Loss**        | 0.01008    | 0.0267    | N/A                      |

🏆 **Best Model:** DUCKNet with the highest **Dice Coefficient (88.72%)** and lowest **Validation Loss (0.01008)**.

---

## 🏁 Conclusion
DUCKNet outperforms traditional segmentation models in tumor boundary detection, making it an excellent choice for MRI-based tumor segmentation.

---

## 🛠 Usage
```bash
git clone https://github.com/suman105/BraTS-Dataset-PolySegmentation.git
cd BraTS-PolypSegmentation
jupyter notebook polysegmentationofbradtsdataset.ipynb
```

---

## 🤝 Team Members
👨‍💻 **Shreyas Prabhakar**  
👨‍💻 **Suman Majjari**  
👨‍💻 **Siva Pavan Inja**  
👨‍💻 **Talha Jabbar**  
👨‍💻 **Aditya Madalla**  

---

## 📚 References
- 📄 [BraTS2020 Dataset](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)
- 📄 [DUCK-Net Paper](https://github.com/RazvanDu/DUCK-Net)

---

## 🤝 Acknowledgments
🔹 **Professor Dr. Victor Sheng** for guidance.  
🔹 **BraTS2020 Dataset Authors** for data contribution.  
🔹 **Texas Tech University** for computational resources.  
🔹 **High-Performance Computing Cluster** for model testing.  

---

## 👤 Author
[Suman Majjari](https://www.github.com/suman105)

---

## 📜 License
This project is based on **DUCK-Net** by Razvan Du et al. and modified under the **MIT License**.

```
@article{ModifiedDUCKNet2024,
  author = {Shreyas Prabhakar, Suman Majjari, Siva Pavan Inja, Talha Jabbar, Aditya Madalla},
  title = {Modified DUCK-Net: Customized Image Segmentation},
  year = {2024},
  note = {Guided by Professor Victor Sheng},
  url = {https://github.com/shrprabh/BraTS-PolypSegmentation}
}
```

📌 **Enjoy coding & contribute to the project! 🚀**

