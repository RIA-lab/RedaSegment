# RedaSegment

![Architecture](RedaSegment.png)

**RedaSegment** is a deep learning model designed to predict enzyme thermal parameters, including optimal temperature, temperature range, and stability.

---

## 🔧 Setup

### 1. Download Model Weights
- [Download RedaSegment weights](https://drive.google.com/file/d/14fKcSSs5HYgLTpO2J_Qy_STen9pJs9gC/view?usp=sharing)
- Unzip the downloaded file and place the contents in the `output/` folder.

### 2. Download ESM-2 Weights
- Download ESM-2 weights manually.
- Place the downloaded weights in the `esm150/` folder.

---

## 🚀 Inference

To run inference on your own CSV file:

```bash
python inference.py \
  --config configs/stability_reda_segment.yaml \
  --weight output/split_clusters_stability_reda_segment/model.safetensors \
  --file path/to/your/csv/file
