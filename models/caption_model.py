# BLIP image captioning wrapper
import torch
import torch.nn.functional as F
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np

class CaptionModel:
    """BLIP image captioning model wrapper"""
    
    def __init__(self, model_name="Salesforce/blip-image-captioning-large", device="cpu"):
        """
        Initialize BLIP model
        
        Args:
            model_name: HuggingFace model identifier
            device: 'cuda' or 'cpu'
        """
        print(f"Loading caption model: {model_name}")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device
        print(f"✓ Caption model loaded on {device}")
    
    def generate_caption(self, image, max_length=50, num_beams=5):
        """
        Generate caption for image
        
        Args:
            image: PIL Image
            max_length: Maximum caption length
            num_beams: Beam search width
            
        Returns:
            Tuple of (caption: str, confidence: float)
        """
        # Preprocess
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_beams=num_beams,
                output_scores=True,
                return_dict_in_generate=True,
            )
        
        # Decode caption
        caption = self.processor.decode(outputs.sequences[0], skip_special_tokens=True).strip()
        
        # Calculate confidence
        confidence = self._calculate_confidence(outputs)
        
        return caption, confidence
    
    def _calculate_confidence(self, outputs):
        """Calculate average token probability as confidence score"""
        if hasattr(outputs, 'scores') and outputs.scores:
            probs = []
            for score in outputs.scores:
                prob = F.softmax(score, dim=-1)
                max_prob = prob.max().item()
                probs.append(max_prob)
            return float(np.mean(probs))

        return 0.75  # Default for beam search
