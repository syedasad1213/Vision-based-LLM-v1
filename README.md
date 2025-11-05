## Features

I add object detection capabilities to the image captioning system:

- Image captioning using BLIP-large model
- Object detection and localization with YOLOv8
- Visual bounding boxes with class labels and confidence scores
- English to German translation
- Configurable detection thresholds

## Project Structure

```
Vision-based-LLM-v1/
├── app.py                      # Main Gradio application
├── models/
│   ├── __init__.py
│   ├── caption_model.py        # BLIP captioning wrapper
│   ├── detection_model.py      # YOLOv8 detection wrapper
│   └── translation_model.py    # MarianMT translation wrapper
├── utils/
│   ├── __init__.py
│   └── image_utils.py          # Image processing utilities
├── requirements.txt
└── README.md
```

## Installation

### Create Virtual Environment

It's recommended to use a virtual environment to avoid package conflicts.

Windows:
```bash
cd week2_object_detection
python -m venv venv
venv\Scripts\activate
```

Linux/Mac:
```bash
cd week2_object_detection
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear in your terminal prompt.

### Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Note: The first run will download approximately 2GB of model files:
- BLIP-large (around 1GB)
- YOLOv8n (around 6MB)
- MarianMT translation model (around 300MB)

### Verify Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)"
python -c "import gradio; print('Gradio:', gradio.__version__)"
```

All three commands should print version numbers without errors.

### Run Application

```bash
python app.py
```

The application will start at http://127.0.0.1:7860

## Usage

### Basic Workflow

1. Upload an image by clicking the upload area or dragging and dropping
2. Click "Analyze Image" button
3. Wait 5-10 seconds for processing
4. Review results:
   - Annotated image with bounding boxes
   - English caption with confidence score
   - German translation
   - List of detected objects with counts

### Settings

Open the Settings accordion to customize behavior:

- **Enable Object Detection** - Toggle object detection on or off
- **Enable Translation** - Toggle German translation on or off
- **Confidence Threshold** - Set minimum detection confidence (0.1 to 0.9)
  - Lower values detect more objects but may include false positives
  - Higher values are more selective but may miss objects

## Model Information

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| Caption | BLIP-large | 1GB | Generate scene descriptions |
| Detection | YOLOv8n | 6MB | Detect and localize objects |
| Translation | opus-mt-en-de | 300MB | Translate English to German |

### YOLOv8 Variants

The default model uses YOLOv8n (nano) for speed. You can change this in `models/detection_model.py` at line 19:

```python
self.model = YOLO("yolov8n.pt")  # Change to other variants
```

Available options:
- yolov8n.pt - Nano, 6MB (fastest, default)
- yolov8s.pt - Small, 22MB
- yolov8m.pt - Medium, 52MB
- yolov8l.pt - Large, 87MB
- yolov8x.pt - Extra Large, 131MB (most accurate)

### Object Classes

YOLOv8 detects 80 object classes from the COCO dataset, including:
- People and animals (person, cat, dog, horse, bird)
- Vehicles (car, truck, bus, motorcycle, bicycle, airplane)
- Indoor objects (chair, couch, tv, laptop, mouse, keyboard)
- Outdoor objects (traffic light, fire hydrant, stop sign)

See the full list at: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml

## Example Output

Input: Street scene with people and cars

```
Caption: people walking on a city street with cars
Confidence: 89.5%
German: Menschen, die auf einer Stadtstraße mit Autos gehen

Objects Detected: 5

Total: 5 objects

• person: 3 (avg conf: 92.3%)
• car: 2 (avg conf: 87.1%)
```

## Troubleshooting

### ModuleNotFoundError: No module named 'ultralytics'

This means the package is installed in a different Python environment than the one running your script.

Solution:
```bash
# Ensure virtual environment is activated
# You should see (venv) in your prompt

# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate

# Then reinstall:
pip install ultralytics
```

### Translation Error with Keras

If you see: "Your currently installed version of Keras is Keras 3..."

Solution:
```bash
pip install tf-keras
```

Alternatively, disable translation in the UI settings.

### Out of Memory Error

Solution 1: Use smaller models

In `models/caption_model.py`, change to:
```python
model_name = "Salesforce/blip-image-captioning-base"
```

In `models/detection_model.py`, the smallest model is already used.

Solution 2: Reduce image size

In `app.py`, add image resizing before analysis:
```python
from utils.image_utils import resize_image
image = resize_image(image, max_size=800)
```

### Slow Performance on CPU

Processing takes 10-20 seconds per image on CPU. This is normal.

To improve speed, use a GPU:
1. Install CUDA-enabled PyTorch from https://pytorch.org/get-started/locally/
2. Models will automatically detect and use GPU

### YOLOv8 Download Fails

Manual download steps:
1. Download from: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
2. Place file in project root: `week2_object_detection/yolov8n.pt`
3. The model will be found automatically

## Customization

### Change Bounding Box Colors

Edit `utils/image_utils.py`:
```python
COLORS = [
    '#FF0000',  # Red
    '#00FF00',  # Green
    '#0000FF',  # Blue
    # Add more colors as needed
]
```

### Train Custom Object Detector

YOLOv8 uses COCO classes by default. To detect custom objects:
1. Train a custom YOLOv8 model (see Ultralytics documentation)
2. Update the model path in `detection_model.py`

### Adjust Caption Quality

Edit `models/caption_model.py`:
```python
outputs = self.model.generate(
    **inputs,
    max_new_tokens=50,
    num_beams=10,  # Increase from 5 for better quality
)
```

Higher beam values improve quality but slow down generation.

## Performance Benchmarks

Test system: Intel i7, 16GB RAM, CPU only

| Image Size | Caption | Detection | Translation | Total |
|------------|---------|-----------|-------------|-------|
| 640x480    | 2.1s    | 1.8s      | 0.4s        | 4.3s  |
| 1024x768   | 2.8s    | 2.4s      | 0.5s        | 5.7s  |
| 1920x1080  | 4.2s    | 3.9s      | 0.6s        | 8.7s  |

With GPU (NVIDIA RTX 3060):
- Overall 3-5x faster
- Caption: 0.5-1s
- Detection: 0.3-0.8s

## Next Steps

I will add vision-language model integration:
- Interactive question and answer about images
- Conversation context tracking
- LLM integration (GPT-4, Claude, or local models)
- Multi-turn dialogue support

## Common Issues

**First run is slow**: Models need to download and cache. Subsequent runs are faster.

**Virtual environment recommended**: Prevents conflicts between different Python projects.

**Confidence threshold**: Start at 0.25 and adjust based on results. Lower values show more detections.

**GPU recommended**: Provides 3-5x speed improvement over CPU.

**Image quality matters**: Clear, well-lit images produce better results.

## Requirements

- Python 3.8 or higher
- Approximately 2GB disk space for models
- 4GB RAM minimum (8GB recommended)
- GPU optional but recommended for better performance

## License

MIT License - Free to use and modify.

## Resources

- YOLOv8 Documentation: https://docs.ultralytics.com/
- BLIP Model: https://huggingface.co/Salesforce/blip-image-captioning-large
- Gradio Documentation: https://www.gradio.app/docs/
- PyTorch Installation: https://pytorch.org/get-started/locally/


