from PIL import Image
from style_transfer import StyleTransfer
import torch
from remove_bg import BackgroundRemoval

# Load content image.
original_image = Image.open("content.png")

content_image=original_image

# Load MODNet and remove content image background.
# background_removal = BackgroundRemoval("./weights/modnet.pth", device="cpu")
# content_image = background_removal.remove_background(
#     img=original_image,
#     bg_color="black",
# )

# Load style images.
style_images = [
    Image.open("style_1.jpg"), 
    Image.open("style_2.jpg"),
]

# Load and run style transfer module.
st = StyleTransfer(device=torch.device("cuda"), pooling="max")

result_image = st.stylize(
    content_image=content_image, 
    style_images=style_images,
    content_weight=0.05,
    face_weight=0.25,
)

# Save result to disk.
result_image.save("out.png")