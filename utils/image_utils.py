# week2_object_detection/utils/image_utils.py
# Image processing utilities for Week 2

from PIL import Image, ImageDraw, ImageFont
import random

# Color palette for bounding boxes
COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
    '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788'
]

def draw_detections(image, detections, show_confidence=True):
    """
    Draw bounding boxes on image
    
    Args:
        image: PIL Image
        detections: List of detection dicts with format:
                   [{'class': 'person', 'confidence': 0.95, 'bbox': [x1,y1,x2,y2]}, ...]
        show_confidence: Whether to show confidence scores
        
    Returns:
        PIL Image with bounding boxes drawn
    """
    # Create a copy to avoid modifying original
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # Try to load a nice font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
    
    # Draw each detection
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        class_name = det['class']
        confidence = det['confidence']
        
        # Choose color based on class (consistent colors for same class)
        color = COLORS[hash(class_name) % len(COLORS)]
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Prepare label
        if show_confidence:
            label = f"{class_name} {confidence:.2f}"
        else:
            label = class_name
        
        # Get text bounding box for background
        try:
            bbox = draw.textbbox((x1, y1 - 25), label, font=font)
        except:
            # Fallback for older PIL versions
            bbox = [x1, y1 - 25, x1 + len(label) * 8, y1 - 5]
        
        # Draw label background
        draw.rectangle(bbox, fill=color)
        
        # Draw label text
        draw.text((x1, y1 - 25), label, fill='white', font=font)
    
    return img_draw

def format_detections(detections):
    """
    Format detection results as readable text
    
    Args:
        detections: List of detection dicts
        
    Returns:
        Formatted string with detection summary
    """
    if not detections:
        return "No objects detected."
    
    # Group by class
    class_counts = {}
    for det in detections:
        class_name = det['class']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Build output
    lines = []
    lines.append(f"**Total: {len(detections)} objects**\n")
    
    # Sort by count (descending)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    for class_name, count in sorted_classes:
        # Get confidence scores for this class
        confidences = [
            det['confidence'] for det in detections 
            if det['class'] == class_name
        ]
        avg_conf = sum(confidences) / len(confidences)
        
        lines.append(f"• **{class_name}**: {count} (avg conf: {avg_conf:.1%})")
    
    return "\n".join(lines)

def resize_image(image, max_size=1024):
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: PIL Image
        max_size: Maximum dimension (width or height)
        
    Returns:
        Resized PIL Image
    """
    width, height = image.size
    
    if max(width, height) <= max_size:
        return image
    
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)