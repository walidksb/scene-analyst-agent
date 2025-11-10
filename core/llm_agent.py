from core.vision_encoder import VisionLanguageModel

class SceneReasoner:
    """
    Offline reasoning agent that uses BLIP-2 for visual question answering
    and optionally augments reasoning with object detections.
    """
    def __init__(self):
        self.vlm = VisionLanguageModel()

    def reason_about_scene(self, image_path: str, question: str, detected_objects=None) -> str:
        """
        Generate an answer to a visual question based on an image and optional context.
        """
        context = ""
        if detected_objects:
            context = f"The following objects were detected: {', '.join(detected_objects)}. "

        combined_question = context + question
        answer = self.vlm.answer_question(image_path, combined_question)
        return answer
