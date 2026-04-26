# 📸 Automated Photo Enhancement with Tone Curves

A learning-based system for automatic photo enhancement by predicting **global tone curves** for luminance and color adjustment.

---

## ✨ Overview

This project learns aesthetic photo adjustments from expert-retouched images.

Instead of directly generating pixels, the model predicts **interpretable tone curves**, enabling both:

* automatic enhancement
* user-controllable editing

---

## 🧠 Method

* **Backbone**: EfficientNet-B0 for aesthetic feature extraction

* **Prediction head**: Lightweight MLP outputs:

  * Luma tone curve (global brightness)
  * RGB tone curves (color adjustment)

* **Rendering pipeline**:

  * Differentiable image processing using piecewise-linear LUTs

* **Training**:

  * Indirect supervision from input–retouched pairs
  * Learns editing parameters instead of pixel mappings

---

## 📊 Dataset

* Adobe FiveK
* RAW / expert-retouched pairs (Expert C)
* Images resized to `224×224`

---

## 📈 Results

* **PSNR**: ~23 dB
* **SSIM**: 0.83

The model learns:

* Exposure correction
* Contrast enhancement
* Color balancing

---

## 🖼️ Visual Results

**Input**
![Input Image](images/example_input.png)

**Expert Retouch (Target)**
![Target Image](images/example_target.png)

**Predicted Tone Curves**
![Predicted Curve](images/example_curve.png)

**Model Output**
![Model Output](images/example_output.png)

---

## 🔍 What the Model Learns

### 🌗 Luma Curve (Brightness & Contrast)

* Curve slightly **above identity in mid-tones**
  → Brightens darker regions (e.g., interior details)

* Highlights are **slightly compressed**
  → Avoids overexposure in bright areas

**Effect**

* Better dynamic range perception
* Improved visibility without highlight clipping

---

### 🎨 RGB Color Curves (Color Balance)

* **Blue channel elevated**
  → Reduces yellowish cast (cooler tones)

* **Green moderately increased**
  → Stabilizes natural color balance

* **Red suppressed in shadows, boosted in highlights**
  → Keeps warmth without oversaturation

**Effect**

* Corrected color cast
* More balanced, visually pleasing tones

---

### 🧠 Overall Behavior

The model performs a structured transformation:

* Adjusts **global illumination** via Luma
* Refines **color distribution** via RGB curves
* Mimics expert editing in a **low-dimensional, interpretable space**

This leads to:

* good generalization
* stable outputs
* user controllability

---

### 🎯 Key Insight

The model does not generate a new image.
It **learns how to edit an image**, similar to adjusting sliders in Lightroom.

---

## 🌐 Web Demo

👉 [https://dauduchieu.github.io/literoom/](https://dauduchieu.github.io/literoom/)

* Fully client-side (TensorFlow.js)
* No server required
* Real-time inference in browser

**Expert Reference**
![Expert Reference](images/web_demo_expert.png)

**Web Interface**
![Web Demo Overview](images/web_demo.png)

---

## 🎛️ Interactive Editing

Users can refine the output via:

* Brightness (Luma)
* Red / Green / Blue channels

→ Maintains **interpretability + controllability** vs black-box models

---

## 🚀 Future Work

* Local tone mapping (region-aware enhancement)
* Export 3D LUTs for Lightroom / Photoshop
* Mobile optimization

---

## 🛠️ Tech Stack

* TensorFlow / Keras
* EfficientNet-B0
* Differentiable image processing
* TensorFlow.js

---

## 🔗 Repository

👉 [https://github.com/dauduchieu/literoom](https://github.com/dauduchieu/literoom)

---
