import os

from PIL import Image, ImageFile

from lavis.datasets.datasets.image_text_pair_datasets import ImageTextPairDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ShutterStock(ImageTextPairDataset):

    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]
        img_id = int(ann["image"].split("/")[-1].split(".jpg")[0])

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])

        return {"image": image, "image_id": img_id, "text_input": caption}
