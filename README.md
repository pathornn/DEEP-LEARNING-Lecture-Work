# DEEP-LEARNING-Lecture-Work

คลังรวบรวมแบบฝึกหัดปฏิบัติการและโครงงานบูรณาการ (Capstone Project) สำหรับการเรียนรู้ด้านการเรียนรู้เชิงลึก (Deep Learning) และการมองเห็นของคอมพิวเตอร์ (Computer Vision) ด้วยภาษา Python, ไลบรารี NumPy, Matplotlib และเฟรมเวิร์ก TensorFlow / Keras โดยครอบคลุมเนื้อหาตั้งแต่พื้นฐานโครงข่ายประสาทเทียม การปรับปรุงประสิทธิภาพโมเดล การตรวจจับวัตถุ ไปจนถึงการแบ่งส่วนภาพระดับพิกเซล

---

## สารบัญ (Table of Contents)
- [ภาพรวมของวิชา (Course Overview)](#ภาพรวมของวิชา-course-overview)
- [เครื่องมือและสภาพแวดล้อมที่ใช้ (Environment & Tools)](#เครื่องมือและสภาพแวดล้อมที่ใช้-environment--tools)
- [โครงสร้างของคลังข้อมูล (Repository Structure)](#โครงสร้างของคลังข้อมูล-repository-structure)
- [รายละเอียดแบบฝึกหัด (Lecture Works)](#รายละเอียดแบบฝึกหัด-lecture-works)
  - [Work 01: Computer Vision Fundamentals & Basic ANN](#work-01-computer-vision-fundamentals--basic-ann-chapters-12)
  - [Work 02: Optimization, CNN & Regularization](#work-02-optimization-cnn--regularization-chapters-34)
  - [Work 03: Multiscale Analysis, Normalization & ResNet](#work-03-multiscale-analysis-normalization--resnet-chapters-56)
  - [Work 04: Object Detection & Instance Segmentation](#work-04-object-detection--instance-segmentation-chapters-78)
- [โครงงานบูรณาการ (Capstone Project)](#โครงงานบูรณาการ-capstone-project)
- [ตารางอ้างอิงการเลือก Activation และ Loss Function](#ตารางอ้างอิงการเลือก-activation-และ-loss-function)

---

## ภาพรวมของวิชา (Course Overview)
วิชานี้เน้นการทำความเข้าใจความแตกต่างระหว่างการมองเห็นของมนุษย์ (ชีววิทยาและฟังก์ชันสมอง เช่น การตอบสนองของเซลล์ตามงานวิจัยของ Hubel & Wiesel, 1959) กับมุมมองของคอมพิวเตอร์ที่เห็นข้อมูลภาพเป็นตารางตัวเลข (Pixel Matrix) โดยศึกษาวิวัฒนาการจากการสกัดฟีเจอร์พื้นฐาน (Features เช่น ขอบ มุม เส้นตรง) ไปสู่การจำแนกประเภท (Classification), การตรวจจับวัตถุ (Object Detection) และการแบ่งส่วนภาพ (Instance Segmentation)

---

## เครื่องมือและสภาพแวดล้อมที่ใช้ (Environment & Tools)
- **Programming Language:** Python (3.10 – 3.12)
- **Development Environment:** Jupyter Notebook / JupyterLab, Google Colab หรือ PyCharm Community
- **Core Libraries:**
  - `NumPy`: จัดการเมทริกซ์และข้อมูลอาร์เรย์หลายมิติ
  - `Matplotlib`: การแสดงผลภาพ พล็อตกราฟ และแดชบอร์ดสรุปผล
  - `TensorFlow` & `Keras`: การสร้าง ฝึกสอน และปรับแต่งโครงข่ายประสาทเทียม
  - `TensorFlow Datasets (tfds)`: โหลดและจัดการแบ่งชุดข้อมูล (Dataset Slicing)

---

## โครงสร้างของคลังข้อมูล (Repository Structure)

```text
DEEP-LEARNING-Lecture-Work/
├── README.md
├── Work01_Image_Processing_and_Basic_ANN/
│   ├── Ch1_NumPy_Matplotlib_Features.ipynb
│   └── Ch2_Fashion_MNIST_ANN_Classifier.ipynb
├── Work02_Optimization_and_Deep_CNN/
│   ├── Ch3_Loss_Optimizers_BatchSize_Lab.ipynb
│   └── Ch4_Deep_CNN_Dropout_Regularization.ipynb
├── Work03_Multiscale_BatchNorm_and_ResNet/
│   ├── Ch5_Pooling_and_BatchNormalization.ipynb
│   └── Ch6_Vanishing_Gradient_and_ResNet_Transfer.ipynb
├── Work04_Object_Detection_and_Segmentation_Tech/
│   ├── Ch7_IoU_NMS_and_mAP_Evaluation.ipynb
│   └── Ch8_MultiLabel_Augmentation_and_Contours.ipynb
└── Capstone_Project_Visual_Recognition_System/
    ├── 01_data_pipeline_and_contour_mask.py
    ├── 02_custom_multiscale_cnn_and_resnet.py
    ├── 03_two_phase_trainer_and_grad_inspector.py
    ├── 04_roi_extraction_and_nms_engine.py
    └── 05_evaluation_metrics_and_dashboard.ipynb
```

---

## รายละเอียดแบบฝึกหัด (Lecture Works)

### Work 01: Computer Vision Fundamentals & Basic ANN (Chapters 1–2)
- **Part 1: การจัดการ Pixel Matrix และแสดงผลข้อมูลภาพ (Chapter 1)**
  - การใช้งาน `np.ndarray`, การเลือกชนิดข้อมูล (`dtype`) เช่น `np.uint8`, `np.float32`, `np.int32`
  - การตรวจสอบและแปลงมิติข้อมูล: `.ndim`, `.shape`, `reshape()`, `resize()`, `np.append()` และ `np.expand_dims()`
  - การทำ Data Slicing แบบ 1 มิติและ 2 มิติ (เช่น `Q[1:4, 2:6]`) เพื่อเลือกพื้นที่สนใจ
  - การพล็อตกราฟ Line, Scatter, Bar, Histogram และการแสดงภาพด้วย `Matplotlib`
- **Part 2: การจำแนกประเภทด้วย Feedforward ANN บน Fashion MNIST (Chapter 2)**
  - การโหลดและสำรวจชุดข้อมูล Fashion MNIST (Train 60,000 ภาพ, Test 10,000 ภาพ ขนาด $28\times28$ พิกเซล Grayscale)
  - การแบ่งสัดส่วนชุดข้อมูล (Train / Validation / Test) และการทำ Normalisation ด้วยการหาร `255.0`
  - การคำนวณ Sum of Product: $O_k = (1)b_k + \sum_i w_{k,i} x_i$
  - การสร้างโมเดล `keras.Sequential`: ชั้น `Flatten(input_shape=(28, 28))` และ `Dense(10, activation=tf.nn.softmax)`
  - พิสูจน์ที่มาของพารามิเตอร์ 7,850 ตัว ($[784 \times 10] + 10 \text{ Bias}$)
  - Compile โมเดลด้วย `Adam`, `sparse_categorical_crossentropy` และวัดผลด้วย `accuracy`

### Work 02: Optimization, CNN & Regularization (Chapters 3–4)
- **Part 1: ฟังก์ชันวัดความคลาดเคลื่อนและการเพิ่มประสิทธิภาพ (Chapter 3)**
  - การคำนวณ Mean Squared Error (MSE) สำหรับงาน Regression:
    $$J(\theta_1, \theta_2) = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$
  - การทำงานของ Softmax และ Cross-Entropy สำหรับงาน Classification:
    $$D(S, L) = -\sum_i L_i \log(S_i)$$
  - การทำงานของ Gradient Descent, Backpropagation (Chain Rule) และผลกระทบของ Learning Rate ($\alpha$)
  - การหลีกเลี่ยง Local Minimum สู่ Global Minimum ด้วย Optimizer เช่น `Adam` และ `Gradient Descent with Nesterov momentum`
  - เปรียบเทียบความเร็วและการใช้หน่วยความจำ: Batch GD, Stochastic GD (SGD, $\text{Batch}=1, 2$) และ Mini-batch GD ($\text{Batch}=32, 64, 128$)
- **Part 2: โครงข่าย Deep Learning, CNN และการแก้ Overfitting (Chapter 4)**
  - การสร้างโครงข่ายหลายชั้น (Hidden Layers $\ge 2$) และเปรียบเทียบ Activation Function ระหว่าง `ReLU` ($f(x)=\max(0, x)$) กับ `Softplus`
  - การทำงานของ Filter/Kernel ขนาด $3\times3$ ในการกวาดภาพเพื่อสกัด Feature Maps
  - การตรวจสอบประวัติการฝึก (`loss`, `val_loss`, `accuracy`, `val_accuracy`) เพื่อระบุ Underfitting และ Overfitting
  - การแก้ปัญหา Overfitting ด้วย `Dropout` และ `SpatialDropout2D` (สุ่มปิดฟิลเตอร์ยกแผ่น)

### Work 03: Multiscale Analysis, Normalization & ResNet (Chapters 5–6)
- **Part 1: Multiscale Analysis และการปรับสมดุลสถิติ (Chapter 5)**
  - ปัญหาขนาดวัตถุที่ไม่เท่ากันและการใช้ Pooling จำลองการรับรู้ของมนุษย์
  - เปรียบเทียบ Max Pooling กับ Average (Mean) Pooling ด้วย Filter ขนาด $2\times2$ และ Stride $(2, 2)$
  - การออกแบบโดยเพิ่มจำนวน Filter ใน Convolutional Layer ถัดไปเป็น 2 เท่าเพื่อชดเชยการลดขนาดข้อมูล
  - การใช้งาน `Batch Normalization` (`axis=-1`, Zero Mean, $\text{SD}=1$) วางคั่นหลัง Conv/Dense และก่อน Activation
  - การแก้ปัญหากราฟ `val_accuracy` แกว่งด้วยการขยายขนาด `batch_size` (เช่น จาก 128 เป็น 512)
  - ความแตกต่างของมิติการคำนวณ: Batch Norm, Layer Norm, Instance Norm และ Group Norm
- **Part 2: การแก้ปัญหา Vanishing Gradient ด้วย Skip Connections และ ResNet (Chapter 6)**
  - สาเหตุของ Vanishing Gradient จากฟังก์ชัน Sigmoid/Tanh ร่วมกับ Chain Rule
  - การตรวจสอบค่าน้ำหนักที่หยุดนิ่งผ่าน `model.trainable_variables`
  - โครงสร้าง Residual Block และสมการ Skip Connection:
    $$f(x) = g(x) + x$$
  - การทำ Transfer Learning ด้วยสถาปัตยกรรม ResNet (Pre-trained บน ImageNet, ตัดส่วนหัวด้วย `include_top=False`)
  - การรวมฟีเจอร์ด้วย `GlobalAveragePooling2D` หรือ `GlobalMaxPooling2D`
  - การเปรียบเทียบผลลัพธ์ระหว่าง Freeze Mode (`trainable=False`) และ Unfreeze Mode (`trainable=True` บางส่วน)

### Work 04: Object Detection & Instance Segmentation (Chapters 7–8)
- **Part 1: ระบบการตรวจจับวัตถุและตัวชี้วัดความแม่นยำ (Chapter 7)**
  - วิวัฒนาการ Two-stage Detectors: R-CNN, Fast R-CNN (RoI Pooling) และ Faster R-CNN (RPN บน GPU) เทียบกับ Single-stage Detectors
  - การเขียนฟังก์ชันคำนวณ Intersection Over Union (IoU):
    $$IoU(P, G) = \frac{|P \cap G|}{|P \cup G|}$$
  - การทำงานของ Non-Max Suppression (NMS) เพื่อตัดกล่องซ้อนทับ
  - การคำนวณสถานะผลลัพธ์: True Positive (TP), False Positive (FP), False Negative (FN), True Negative (TN)
  - การประเมินผลด้วย Precision, Recall, F1-Score, กราฟ Precision-Recall Curve (11-point interpolated precision), AP และ mAP ($AP@[.50:.95]$)
- **Part 2: การแบ่งส่วนภาพ, การตั้งค่า Output Layer และเทคนิคจัดการข้อมูล (Chapter 8)**
  - นิยาม Instance Segmentation และการเก็บพิกัดขอบแบบ Contour เทียบกับ Binary Mask ในโมเดล Mask R-CNN (RoIAlign)
  - การตัดแบ่งข้อมูลแบบ Slicing ผ่าน `tfds.load(split=...)` เช่น `split=['train[:200]']` หรือ `split=['train[:50%]']`
  - การตั้งค่าโมเดลตามประเภทโจทย์: Binary Classification, Multi-class Single-label, และ Multi-class Multi-label (Multi-hot Encoding)
  - การทำ Data Augmentation ด้วย 3 แนวทาง: `tf.keras.layers`, `tf.image` (Custom Mapping) และ `ImageDataGenerator`

---

## โครงงานบูรณาการ (Capstone Project)
**ชื่อโครงงาน:** ระบบวิเคราะห์ภาพหลายระดับความละเอียด จำแนกหลายคุณลักษณะ และประเมินผลการตรวจจับวัตถุครบวงจร (*End-to-End Multi-Scale Visual Recognition, Multi-Label Classification & Detection Evaluation System*)

### สรุปสาระสำคัญของแต่ละโมดูล:
1. **Module 1: Data & Mask Engineering Pipeline (`01_data_pipeline_and_contour_mask.py`)**
   - โหลดชุดข้อมูลด้วย `tfds.load(split=...)` จัดการเมทริกซ์พิกเซลด้วย NumPy (`dtype=np.uint8` แปลงเป็น `np.float32` พร้อมสเกลค่าพิกเซลด้วยการหาร `255.0`)
   - แปลงข้อมูลพิกัด Contour $(x, y)$ ให้เป็น Segmentation Mask และสกัดพิกัด Bounding Box $(x, y, w, h)$
   - สร้างไปป์ไลน์ Data Augmentation บน `tf.data.Dataset` ร่วมกับ `tf.image` และเตรียม Label แบบ Multi-hot
2. **Module 2: Dual Architecture & Diagnostic Inspector (`02_custom_multiscale_cnn_and_resnet.py`)**
   - **Model A (Custom Multiscale CNN):** โครงข่ายที่ใช้ Filter $3\times3$, มีการทำ Pooling ($2\times2$) สลับกับการคูณ 2 เท่าของจำนวน Filter, คั่นด้วย Batch Norm (`axis=-1`), Activation `ReLU`/`Softplus` และใส่ Regularization ด้วย `SpatialDropout2D`
   - **Model B (ResNet Transfer Learning):** โมเดล ResNet ที่มี Skip Connection ($f(x) = g(x) + x$) ต่อกับ `GlobalAveragePooling2D` และเลเยอร์ Output หลายโหนดด้วย Sigmoid สำหรับโจทย์ Multi-label
   - **Diagnostics:** ฟังก์ชันตรวจสอบค่า Gradient และค่าน้ำหนักผ่าน `model.trainable_variables` เพื่อตรวจวัดอาการ Vanishing Gradient
3. **Module 3: Two-Phase Training Suite (`03_two_phase_trainer_and_grad_inspector.py`)**
   - ฝึกสอนแบบ 2 ระยะ: Phase 1 (Freeze Base Model) และ Phase 2 (Unfreeze เฉพาะเลเยอร์ท้ายสุด พร้อมปรับลด Learning Rate $\alpha$)
   - ทำการทดสอบเปรียบเทียบ Optimizer (`Adam`, `Nesterov momentum`, `SGD`) และเปรียบเทียบขนาด Batch Size (32, 128, 512) ต่อความนิ่งของกราฟการฝึก
4. **Module 4: RoI Feature Extraction & Custom NMS (`04_roi_extraction_and_nms_engine.py`)**
   - สกัด Feature Maps จากชั้น Convolutional มาตัดส่วนพื้นที่เป้าหมายตามแนวคิด RoI
   - พัฒนาอัลกอริทึมคำนวณ IoU และ Non-Max Suppression (NMS) ด้วยตนเองเพื่อคัดกรองกรอบวัตถุ
5. **Module 5: Evaluation Dashboard (`05_evaluation_metrics_and_dashboard.ipynb`)**
   - พล็อตกราฟประวัติการฝึก (`loss`, `val_loss`, `accuracy`, `val_accuracy`)
   - แสดงผลภาพ Feature Maps, Mask จาก Contour และกรอบ Bounding Box หลัง NMS
   - คำนวณตารางเมทริกซ์ TP, FP, FN, กราฟ 11-point Precision-Recall Curve, AP และคำนวณสรุปค่า mAP ($AP@[.50:.95]$)

---

## ตารางอ้างอิงการเลือก Activation และ Loss Function

| Problem Type | Last-layer Output Nodes | Hidden-layer Activation | Last-layer Activation | Loss Function |
| :--- | :---: | :---: | :---: | :--- |
| **Binary classification** | 1 | ReLU / Softplus | Sigmoid | Binary Crossentropy |
| **Multi-class, single-label classification** | Number of classes | ReLU / Softplus | Softmax | Categorical Crossentropy (หรือ Sparse Categorical Crossentropy) |
| **Multi-class, multi-label classification** | Number of classes | ReLU / Softplus | Sigmoid (one for each class) | Binary Crossentropy (รองรับ `from_logits=True`) |