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

### 🔍 What the Model Learns (Interpreting the Tone Curves)

The predicted tone curves reveal how the model adjusts the image:

#### 🌗 Luma Curve (Brightness & Contrast)

* The curve is slightly **above the identity line in mid-tones**
  → Brightens darker regions (e.g., car interior becomes more visible)

* Highlights are **compressed slightly**
  → Prevents overexposure in bright areas (background sky / road)

👉 Effect:

* Improves **dynamic range perception**
* Enhances visibility without blowing out highlights

---

#### 🎨 RGB Color Curves (Color Balance)

* **Blue channel is elevated**
  → Adds cooler tones, reducing yellowish cast

* **Green channel moderately increased**
  → Stabilizes natural colors

* **Red channel suppressed in shadows, increased in highlights**
  → Preserves warmth without oversaturating dark regions

👉 Effect:

* Corrects **color cast**
* Produces a more **cinematic, balanced color tone**

---

#### 🧠 Overall Behavior

Instead of directly modifying pixels, the model learns a structured transformation:

* Adjusts **global illumination (exposure + contrast)** via Luma
* Refines **color distribution** via channel-wise curves
* Mimics expert editing in a **low-dimensional, interpretable space**

This explains why the model:

* Generalizes well
* Produces stable results
* Remains controllable by users

---

### 🎯 Key Insight

The model does not "generate" a new image —
it **learns how to edit an image**, similar to how a human would adjust sliders in Lightroom.

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
