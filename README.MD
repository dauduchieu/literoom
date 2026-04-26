# 📸 Automated Photo Enhancement with Tone Curves

A learning-based system for automatic photo enhancement by predicting **global tone curves** for luminance and color adjustment.

---

## ✨ Overview

This project aims to enhance images by learning aesthetic adjustments from expert-retouched photos.
Instead of directly generating images, the model predicts **interpretable tone curves**, enabling both automation and user control.

---

## 🧠 Method

* Backbone: **EfficientNet-B0** for aesthetic feature extraction

* A lightweight **MLP head** predicts:

  * **Luma tone curve** (global brightness)
  * **RGB tone curves** (color adjustment)

* A fully **differentiable image processing pipeline** applies the curves using piecewise-linear LUTs

* Training strategy:

  * **Indirect supervision** using input–retouched image pairs
  * The model learns to infer editing parameters rather than pixel-wise outputs

---

## 📊 Dataset

* **Adobe FiveK**
* Uses RAW / expert-retouched pairs (Expert C)
* Images resized to `224×224` for training

---

## 📈 Results

* Validation performance:

  * **PSNR**: ~23 dB
  * **SSIM**: 0.83

The model learns natural adjustments including:

* Exposure correction
* Contrast enhancement
* Color balancing

---

## 🖼️ Visual Results

### Input

![Input Image](images/example_input.png)

### Expert Retouch (Target)

![Target Image](images/example_target.png)

### Predicted Tone Curves

![Predicted Curve](images/example_curve.png)

### Model Output

![Model Output](images/example_output.png)

---

## 🌐 Web Demo

👉 Try it here: **[https://dauduchieu.github.io/literoom/](https://dauduchieu.github.io/literoom/)**

* Runs entirely **client-side (TensorFlow.js)**
* No server required
* Real-time inference in the browser

### Demo Preview

**Expert Reference**
![Expert Reference](images/web_demo_expert.png)

**Web Interface**
![Web Demo Overview](images/web_demo.png)

**User Refinement**
![User Refinement](images/web_demo_user_refinement.png)

---

## 🎛️ Interactive Editing

After prediction, users can manually refine the output via:

* Brightness (Luma)
* Red / Green / Blue channels

This preserves **interpretability + controllability**, unlike black-box enhancement models.

---

## 🚀 Future Work

* Local tone mapping (region-aware enhancement)
* Export **3D LUTs** for Lightroom / Photoshop
* Model optimization for mobile deployment

---

## 🛠️ Tech Stack

* TensorFlow / Keras
* EfficientNet-B0
* Differentiable image processing
* TensorFlow.js (deployment)

---

## 🔗 Repository

👉 [https://github.com/dauduchieu/literoom](https://github.com/dauduchieu/literoom)

---
