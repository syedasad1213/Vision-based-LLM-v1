import torch
from transformers import MarianMTModel, MarianTokenizer

class TranslationModel:
    """MarianMT English to German translation wrapper"""
    
    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-de", device="cpu"):
        """
        Initialize translation model
        
        Args:
            model_name: HuggingFace model identifier
            device: 'cuda' or 'cpu'
        """
        print(f"Loading translation model: {model_name}")
        try:
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            self.device = device
            self.available = True
            print(f"✓ Translation model loaded on {device}")
        except Exception as e:
            print(f"✗ Translation model failed: {e}")
            self.available = False
    
    def translate(self, text, max_length=512):
        """
        Translate English text to German
        
        Args:
            text: English text
            max_length: Maximum output length
            
        Returns:
            German translation string
        """
        if not self.available:
            return "[Translation not available]"
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # Generate translation
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=max_length)
            
            # Decode
            translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated
        
        except Exception as e:
            print(f"Translation error: {e}")

            return f"[Translation error: {str(e)}]"
