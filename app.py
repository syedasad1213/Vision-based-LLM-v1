import gradio as gr
from models.caption_model import CaptionModel
from models.detection_model import DetectionModel
from models.translation_model import TranslationModel
from utils.image_utils import draw_detections, format_detections
import torch

# Initialize models
print("Loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"

caption_model = CaptionModel(device=device)
detection_model = DetectionModel(device=device)
translation_model = TranslationModel(device=device)

print("All models loaded!")

def analyze_image(image, detect_objects=True, translate=True, confidence_threshold=0.25):
    """
    Comprehensive image analysis
    
    Args:
        image: PIL Image
        detect_objects: Whether to run object detection
        translate: Whether to translate caption to German
        confidence_threshold: Minimum confidence for detections
    """
    if image is None:
        return None, "", 0.0, "", "", "", "Please upload an image."
    
    # 1. Generate caption
    caption, confidence = caption_model.generate_caption(image)
    
    # 2. Translate if requested
    if translate:
        german_caption = translation_model.translate(caption)
    else:
        german_caption = "[Translation disabled]"
    
    # 3. Object detection
    if detect_objects:
        detections = detection_model.detect(image, conf_threshold=confidence_threshold)
        annotated_image = draw_detections(image, detections)
        detection_summary = format_detections(detections)
        object_count = len(detections)
    else:
        annotated_image = image
        detection_summary = "[Object detection disabled]"
        object_count = 0
    
    # 4. Build comprehensive summary
    summary = (
        f"**Caption:** {caption}\n"
        f"**Confidence:** {confidence:.1%}\n"
        f"**German:** {german_caption}\n\n"
        f"**Objects Detected:** {object_count}\n"
        f"{detection_summary}"
    )
    
    return (
        annotated_image,
        caption,
        confidence,
        german_caption,
        detection_summary,
        object_count,
        summary
    )

# Gradio Interface
with gr.Blocks(title="Vision Analysis Agent - Week 2", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Vision Analysis Agent - Object Detection")
    gr.Markdown("""
    **Week 2 Features:**
    - Image Captioning (BLIP)
    - Object Detection (YOLOv8)
    - German Translation
    - Visual Bounding Boxes
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Image", type="pil", height=400)
            
            with gr.Accordion("Settings", open=False):
                enable_detection = gr.Checkbox(label="Enable Object Detection", value=True)
                enable_translation = gr.Checkbox(label="Enable Translation", value=True)
                conf_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.25,
                    step=0.05,
                    label="Detection Confidence Threshold"
                )
            
            analyze_btn = gr.Button("Analyze Image", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            output_image = gr.Image(label="Annotated Image", interactive=False)
            
            with gr.Accordion("Caption", open=True):
                caption_output = gr.Textbox(label="English Caption", lines=2)
                confidence_output = gr.Slider(
                    label="Caption Confidence",
                    minimum=0,
                    maximum=1,
                    interactive=False
                )
            
            with gr.Accordion("Translation", open=False):
                german_output = gr.Textbox(label="German Translation", lines=2)
            
            with gr.Accordion("Detected Objects", open=True):
                detection_output = gr.Textbox(label="Detection Results", lines=6)
                object_count_output = gr.Number(label="Total Objects", precision=0)
            
            with gr.Accordion("Full Summary", open=True):
                summary_output = gr.Textbox(label="Complete Analysis", lines=10)
    
    # Examples
    gr.Examples(
        examples=[
            ["https://images.unsplash.com/photo-1506748686214-e9df14d4d9d0"],
            ["https://images.unsplash.com/photo-1517849845537-4d257902454a"],
        ],
        inputs=input_image
    )
    
    # Connect components
    analyze_btn.click(
        fn=analyze_image,
        inputs=[input_image, enable_detection, enable_translation, conf_slider],
        outputs=[
            output_image,
            caption_output,
            confidence_output,
            german_output,
            detection_output,
            object_count_output,
            summary_output
        ]
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("LAUNCHING WEEK 2: OBJECT DETECTION")
    print("="*60 + "\n")

    demo.launch(share=False)
