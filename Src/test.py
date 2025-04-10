import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
from models import Generator

generator = Generator()
generator.load_state_dict(torch.load("../models/generator.pth"))
generator.eval()

def try_on(person_path, garment_path):
    """ Apply trained model to fit garment onto a person. """
    person_img = to_tensor(Image.open(person_path)).unsqueeze(0)
    garment_img = to_tensor(Image.open(garment_path)).unsqueeze(0)
    
    with torch.no_grad():
        output = generator(person_img, garment_img)
    
    return to_pil_image(output.squeeze(0))

# Example usage
output_img = try_on("../dataset/persons/person1.jpg", "../dataset/garments/garment1.jpg")
output_img.show()
