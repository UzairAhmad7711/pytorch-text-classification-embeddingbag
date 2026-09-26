# PyTorch Text Classification with `nn.EmbeddingBag`

An end-to-end PyTorch text classification project trained on the **AG News Corpus** (120,000+ training samples across 4 categories). This project demonstrates memory-efficient document classification without token padding using PyTorch's `nn.EmbeddingBag` and dynamic batch offsets.

---

## 🚀 Key Features

- **Custom Vocabulary Building:** Built-in numericalization pipeline with frequency thresholding and `<unk>` / `<pad>` handling.
- **Memory-Efficient Architecture:** Utilizes `nn.EmbeddingBag` with `mode="mean"` to reduce variable-length documents into fixed-size summary vectors without computational overhead.
- **Dynamic Batch Offsets (`collate_fn`):** Concatenates multi-length sequence tensors into a 1D sequence using document boundary pointers.
- **Complete Pipeline:** Integrated 20-epoch training, validation metrics, and custom text inference function.

---

## 📊 Dataset

- **Name:** AG News Classification Dataset
- **Classes (4):** `0: World`, `1: Sports`, `2: Business`, `3: Sci/Tech`
- **Training Samples:** 120,000
- **Testing Samples:** 7,600

---

## 🏗️ Model Architecture

| Layer | Type | Details |
| :--- | :--- | :--- |
| **Embedding** | `nn.EmbeddingBag` | `vocab_size` -> `64-dim` (`mode="mean"`) |
| **Output Classifier** | `nn.Linear` | `64-dim` -> `4 classes` |

---

## 📈 Performance Results

- **Training Epochs:** 20
- **Optimizer:** Adam (`lr=0.005`) with StepLR Scheduler
- **Final Test Accuracy:** **89.36%**

---

## 💻 Quickstart & Inference

```python
# Predict custom news headline
sample_text = "NASA launches new rover to explore signs of ancient life on Mars"
predicted_label, confidence = predict_custom_text(sample_text, model, vocab, device)

print(f"Prediction: {predicted_label} | Confidence: {confidence*100:.2f}%")
# Output: Prediction: Sci/Tech | Confidence: 99.8%
