from core.vision_encoder import VisionLanguageModel
from core.scene_tools import ObjectDetector
from core.llm_agent import SceneReasoner

class ScenePipeline:
    def __init__(self):
        self.vlm = VisionLanguageModel()
        self.detector = ObjectDetector()
        self.reasoner = SceneReasoner()

    def analyze(self, image_path: str, question: str):
        print("🔍 Detecting objects...")
        objects = self.detector.detect(image_path)
        print("Objects:", objects)

        print("🧠 Captioning image...")
        caption = self.vlm.caption_image(image_path)

        print("💬 Reasoning about scene...")
        answer = self.reasoner.reason_about_scene(image_path, question, detected_objects=objects)

        print("\n🖼️ Scene Description:", caption)
        print("❓ Question:", question)
        print("💬 Answer:", answer)

        return caption, answer, objects
