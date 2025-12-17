# src/agents/image_agent.py

from tools.image_caption import clip_caption


class ImageAgent:
    def run(self, evidence):
        print("\n[ImageAgent] Generating image captions...")

        updated = []

        for item in evidence:
            fname = item["filename"]

            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                caption = clip_caption(fname)
                item["image_caption"] = caption
            else:
                item["image_caption"] = ""

            updated.append(item)

        return updated
