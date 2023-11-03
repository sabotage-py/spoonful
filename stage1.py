import numpy as np
import torch
from torchvision import models, transforms

device_name = 'cpu'
if torch.cuda.is_available():
    device_name = 'cuda'
device = torch.device(device_name)

MODEL_PATH = 'model_v1.pth'
model = models.resnet18()
model.fc = torch.nn.Linear(512, 3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

mean_array = np.array([0.485, 0.456, 0.406])
std_array = np.array([0.229, 0.224, 0.225])
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean_array, std_array)
])
torch.manual_seed(42)


def get_image_class(img):
    """img: PIL image object.
    Classifies the image and
    returns 0 if img is 'back'
            1 for 'front'
            2 for 'other'
    """
    img = data_transform(img)  # transforms img to 3x224x224 tensor
    img = img.unsqueeze(0)  # convert to 1x3x224x224 tensor
    with torch.no_grad():
        model_output = model(img)
        _, predicted = torch.max(model_output.data, 1)
    return predicted.item()
