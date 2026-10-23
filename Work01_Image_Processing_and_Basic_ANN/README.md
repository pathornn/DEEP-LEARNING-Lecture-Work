# Assignment 1: Computer Vision Fundamentals & Basic Artificial Neural Network (ANN)

* **วิชา:** Deep Learning for Computer Vision
* **เนื้อหาที่ครอบคลุม:** Chapter 1 (Introduction to Computer Vision & Image Data Handling) และ Chapter 2 (Basic Classification with ANN)
* **ไฟล์ที่ต้องส่ง:**
  1. `Ch1_NumPy_Matplotlib_Features.ipynb`
  2. `Ch2_Fashion_MNIST_ANN_Classifier.ipynb`

---

## 1. วัตถุประสงค์การเรียนรู้ (Learning Objectives)

1. เข้าใจความแตกต่างเชิงแนวคิดระหว่างการมองเห็นของสิ่งมีชีวิต (ชีววิทยาของ Hubel & Wiesel, 1959) กับมุมมองของคอมพิวเตอร์ที่ประมวลผลภาพในรูปตารางพิกเซล (Pixel Matrix)
2. จัดการโครงสร้างข้อมูลภาพและควบคุมหน่วยความจำด้วยไลบรารี `NumPy` ผ่านการเลือก `dtype`, การตรวจสอบมิติ (`.ndim`, `.shape`), การแปลงรูปทรงข้อมูล (`reshape()`, `resize()`, `expand_dims()`) และการตัดพื้นที่ภาพ (2D Slicing)
3. แสดงผลภาพถ่ายและการกระจายตัวของข้อมูลด้วยไลบรารี `Matplotlib` (Line, Scatter, Bar, Histogram)
4. เข้าใจหลักการของ Supervised Learning สำหรับงาน Classification และกระบวนการเตรียมข้อมูลภาพ (Data Normalisation, Train/Val/Test Splitting)
5. สร้าง ฝึกสอน และประเมินผล Feedforward Neural Network ด้วย `TensorFlow / Keras` และเข้าใจที่มาทางคณิตศาสตร์ของผลรวมถ่วงน้ำหนัก (Sum of Product) และจำนวนพารามิเตอร์ของโมเดล

---

## 2. รายละเอียดโจทย์ Part 1: การจัดการ Pixel Matrix และแสดงผลข้อมูลภาพ (Chapter 1)
**ไฟล์ส่งงาน:** `Ch1_NumPy_Matplotlib_Features.ipynb`

### Task 1.1: ความเข้าใจพื้นฐานและชนิดข้อมูลของ NumPy (Data Types & Memory Optimization)
* **ข้อกำหนด:**
  1. สร้างเมทริกซ์จำลองภาพขาวดำ (Grayscale) ขนาด $10 \times 10$ พิกเซล โดยสุ่มค่าตัวเลขอิงตามระดับความสว่างจริงของภาพ ($0 - 255$)
  2. กำหนดและเปรียบเทียบชนิดข้อมูล (`dtype`) ระหว่าง `np.uint8`, `np.int32` และ `np.float32` พร้อมอธิบายว่าเหตุใดในระบบ Computer Vision ค่าพิกเซลเริ่มต้นจึงต้องจัดเก็บด้วย `np.uint8` และทำไมจึงต้องแปลงเป็น `np.float32` เมื่อนำเข้าประมวลผลในแบบจำลองทางคณิตศาสตร์
  3. เขียนฟังก์ชันตรวจสอบคุณสมบัติข้อมูล ได้แก่ จำนวนมิติ (`.ndim`), ขนาดมิติ (`.shape`) และขนาดหน่วยความจำ (`.nbytes`)

### Task 1.2: การแปลงรูปทรงและการจัดการมิติข้อมูล (Reshaping & Dimension Expansion)
* **ข้อกำหนด:**
  1. สร้างอาร์เรย์ 1 มิติ ขนาด 784 สมาชิก (จำลองข้อมูลพิกเซลที่แบนราบ)
  2. ใช้คำสั่ง `reshape()` เพื่อแปลงข้อมูลกลับเป็นภาพ 2 มิติขนาด $28 \times 28$ พิกเซล โดยจำนวนสมาชิกรวมต้องเท่าเดิมทุกประการ
  3. สาธิตความแตกต่างระหว่างการใช้ `ndarray.resize()` (In-place method) และ `np.resize()` ในกรณีที่ข้อมูลปลายทางมีขนาดลดลงหรือเพิ่มขึ้น
  4. ใช้คำสั่ง `np.expand_dims()` เพื่อขยายมิติของภาพ $28 \times 28$ ให้กลายเป็นมิติสำหรับรับเข้าโมเดล Deep Learning:
     * ขยายเป็น $28 \times 28 \times 1$ (เพิ่ม Channel ขาวดำ)
     * ขยายเป็น $1 \times 28 \times 28 \times 1$ (เพิ่มมิติ Batch Size สำหรับการป้อนเข้าโมเดล)

### Task 1.3: การตัดเลือกพื้นที่และฟีเจอร์ของภาพ (2D Slicing & Feature Extraction Logic)
* **ข้อกำหนด:**
  1. กำหนดอาร์เรย์ภาพ 2 มิติขนาด $8 \times 8$
  2. ใช้เทคนิค **2D Slicing** ตัดเลือกพื้นที่เฉพาะ เช่น แถวที่ 1 ถึง 3 และคอลัมน์ที่ 2 ถึง 5 (`Q[1:4, 2:6]`)
  3. เขียนอธิบายความเชื่อมโยงเชิงตรรกะว่า การทำ Slicing หรือการเลือกพื้นที่ย่อยนี้ สอดคล้องกับแนวคิดการสกัดคุณลักษณะเฉพาะ (Features) เช่น ขอบ (Edges), มุม (Corners), เส้นตรง ตามงานวิจัยของ Larry Roberts (1963) อย่างไร เพื่อแยกวัตถุออกจากพื้นหลัง

### Task 1.4: การแสดงผลและวิเคราะห์ข้อมูลด้วย Matplotlib
* **ข้อกำหนด:**
  1. นำข้อมูลเมทริกซ์พิกเซลที่สร้างขึ้นมาแสดงผลเป็นภาพด้วย `plt.imshow()` โดยตั้งค่า Color Map ให้เหมาะสม (`cmap='gray'`)
  2. สร้างกราฟแสดงการกระจายตัวของค่าพิกเซล ได้แก่:
     * **Histogram:** แสดงความถี่ของระดับความสว่างในภาพ
     * **Scatter Plot / Bar Plot:** แสดงการแจกแจงของชุดข้อมูลตัวอย่าง
  3. จัดหน้าต่างแสดงผลหลายกราฟในรูปเดียวด้วย `plt.subplot()` พร้อมระบุ Title และ Label ของแต่ละแกนให้ครบถ้วน

---

## 3. รายละเอียดโจทย์ Part 2: การจำแนกประเภทด้วย ANN บนชุดข้อมูล Fashion MNIST (Chapter 2)
**ไฟล์ส่งงาน:** `Ch2_Fashion_MNIST_ANN_Classifier.ipynb`

### Task 2.1: การสำรวจและเตรียมชุดข้อมูล (Data Exploration & Preparation)
* **ข้อกำหนด:**
  1. นำเข้าชุดข้อมูล **Fashion MNIST** โดยตรงจาก `keras.datasets.fashion_mnist`
  2. ตรวจสอบขนาดของข้อมูล:
     * ชุดฝึกสอน (Training Data): ภาพ 60,000 ภาพ ขนาด $28 \times 28$ พิกเซล
     * ชุดทดสอบ (Testing Data): ภาพ 10,000 ภาพ ขนาด $28 \times 28$ พิกเซล
  3. ทำการแบ่งชุดข้อมูลฝึกสอนเพิ่มเติมออกเป็น **Training Set** และ **Validation Set** ในสัดส่วน 80:20 หรือ 70:15:15 เพื่อใช้ในการตรวจสอบผลขณะฝึกสอน
  4. ดำเนินการ **Normalisation** ข้อมูลภาพทั้งหมด โดยนำค่าพิกเซลมาหารด้วย `255.0` (`train_images / 255.0` และ `test_images / 255.0`) เพื่อปรับสเกลค่าให้อยู่ในช่วง $[0.0, 1.0]$ พร้อมอธิบายเหตุผลว่าหากไม่ทำ Normalisation จะส่งผลเสียต่อการปรับค่าน้ำหนัก (Weights Update) อย่างไร

### Task 2.2: การคำนวณทางคณิตศาสตร์และโครงสร้าง Feedforward
* **ข้อกำหนด:**
  1. เขียนคำอธิบายเชิงคณิตศาสตร์ของเลเยอร์ Feedforward ที่โหนดใน Input เชื่อมต่อกับ Output ทุกโหนด:
     $$O_0 = (1)b_0 + w_{0,0}x_0 + w_{0,1}x_1 + \dots + w_{0,783}x_{783}$$
     $$O_1 = (1)b_1 + w_{1,0}x_0 + w_{1,1}x_1 + \dots + w_{1,783}x_{783}$$
     $$\vdots$$
     $$O_9 = (1)b_9 + w_{9,0}x_0 + w_{9,1}x_1 + \dots + w_{9,783}x_{783}$$
  2. อธิบายบทบาทของฟังก์ชัน **Softmax Activation** ที่ใช้ใน Output Layer:
     * การแปลงผลรวมเชิงเส้นให้อยู่ในรูปค่าความน่าจะเป็น (Probability) ช่วง $[0, 1]$
     * ผลรวมของความน่าจะเป็นของทั้ง 10 คลาสต้องเท่ากับ 1 พอดี

### Task 2.3: การสร้างแบบจำลองด้วย Keras Sequential และคำนวณพารามิเตอร์
* **ข้อกำหนด:**
  1. สร้างโมเดลแบบโครงข่ายประสาทเทียมชั้นเดียว (Single-layer ANN) ด้วยคำสั่ง:
     ```python
     import tensorflow as tf
     from tensorflow import keras

     model = keras.Sequential([
         keras.layers.Flatten(input_shape=(28, 28)),
         keras.layers.Dense(10, activation=tf.nn.softmax)
     ])
     ```
  2. รันคำสั่ง `model.summary()` และแสดงการแสดงที่มาของตัวเลขพารามิเตอร์ทั้งหมด **7,850 ตัว** อย่างละเอียด:
     $$\text{Total Parameters} = (784 \text{ Inputs} \times 10 \text{ Outputs}) + 10 \text{ Biases} = 7,840 + 10 = 7,850$$

### Task 2.4: การ Compile, ฝึกสอน (Training) และทำนายผล (Prediction)
* **ข้อกำหนด:**
  1. คอมไพล์โมเดลโดยใช้:
     * **Optimizer:** `Adam()`
     * **Loss Function:** `sparse_categorical_crossentropy`
     * **Metrics:** `['accuracy']`
  2. เทรนโมเดลด้วย `model.fit()` โดยกำหนดจำนวน `epochs=10` และส่งข้อมูล `validation_data` เข้าไปด้วย
  3. สังเกตและบันทึกแนวโน้มของค่า Loss (ต้องลดลง) และ Accuracy (ต้องเพิ่มขึ้น) ในแต่ละรอบการฝึก
  4. นำโมเดลที่ฝึกสอนเสร็จแล้วไปทำนายชุดข้อมูลทดสอบด้วยคำสั่ง:
     ```python
     predictions = model.predict(test_images)
     ```
  5. ใช้ฟังก์ชัน `np.argmax()` ดึงค่าคลาสที่มีความน่าจะเป็นสูงสุด เปรียบเทียบกับป้ายเฉลยจริง (`test_labels`) และใช้ `Matplotlib` สุ่มแสดงภาพ 5 ภาพ พร้อมระบุคลาสที่ทำนายได้เทียบกับเฉลย

---

## 4. เกณฑ์การให้คะแนน (Evaluation Rubric)

| หัวข้อการประเมิน | รายละเอียดการตรวจวัด | คะแนนเต็ม |
|---|---|:---:|
| **Part 1.1 - 1.2: NumPy Operations** | กำหนด `dtype` ถูกต้อง, ใช้ `reshape`, `resize`, `expand_dims` ได้ถูกต้องตามขนาดมิติที่ระบุ | 20 |
| **Part 1.3 - 1.4: Slicing & Visualization** | ทำ 2D Slicing ถูกต้อง, พล็อตภาพและกราฟ Histogram/Bar ด้วย Matplotlib ครบถ้วน | 20 |
| **Part 2.1: Data Preprocessing** | โหลดชุดข้อมูล Fashion MNIST, แบ่ง Train/Val/Test และทำ Normalisation หาร 255.0 ครบถ้วน | 15 |
| **Part 2.2 - 2.3: ANN Architecture & Math** | อธิบายสมการ Sum of Product, Softmax, และแสดงที่มาของ 7,850 พารามิเตอร์ได้ถูกต้อง | 20 |
| **Part 2.4: Training & Evaluation** | คอมไพล์โมเดลถูกประเภท, Loss ลดลงและ Accuracy เพิ่มขึ้น, ทำนายผลและพล็อตเปรียบเทียบเฉลยได้ | 25 |
| **รวมคะแนนทั้งสิ้น** | | **100** |

---

## 5. คำแนะนำในการส่งงาน (Submission Guidelines)
* รันโค้ดทุก Cell ในไฟล์ `.ipynb` ให้ปรากฏ Output แสดงผลลัพธ์และรูปภาพกราฟให้เรียบร้อยก่อน Commit ขึ้น GitHub
* บันทึกคำอธิบายเชิงทฤษฎีใน Text Cell (Markdown) ให้กระชับ สื่อความหมายชัดเจน และไม่เกินขอบเขตของเนื้อหาที่เรียน