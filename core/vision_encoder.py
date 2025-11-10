from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

class VisionLanguageModel:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)

    def caption_image(self, image_path: str) -> str:
        """Generate a caption for the image."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs)
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption

    def answer_question(self, image_path: str, question: str) -> str:
        """Approximate Q&A using the caption + question as prompt (BLIP is not true VQA)."""
        image = Image.open(image_path).convert("RGB")
        prompt = f"Question: {question} Answer:"
        inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=50)
        answer = self.processor.decode(output[0], skip_special_tokens=True)
        return answer
