from torchvision import transforms as ts
import torchvision.models as models
from PIL import Image


class ImageEmbedder:
    def __int__(self):
        self.normalize = ts.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.model = models.squeezenet1_0(pretrained=True, progress=False)

    def embed(self, image_filename):
        image = Image.open(image_filename).convert("RGB")
        image = ts.Resize(224)(image)
        image = ts.CenterCrop(224)(image)
        tensor = ts.ToTensor()(image)
        tensor = self.normalize(tensor).reshape(1, 3, 224, 224)
        vector = self.model(tensor).cpu().detach().numpy().flatten()
        return vector

