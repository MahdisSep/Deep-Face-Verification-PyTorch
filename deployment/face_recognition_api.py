import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.nn.functional import pairwise_distance
from fastapi import FastAPI, UploadFile, File
import uvicorn
import torch
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import numpy as np
import io
import os
from torch import nn
import torch.nn.functional as F
from fastapi.responses import JSONResponse
import os
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(1, 4, kernel_size=3), nn.ReLU(inplace=True), nn.BatchNorm2d(4),
            nn.ReflectionPad2d(1), nn.Conv2d(4, 8, kernel_size=3), nn.ReLU(inplace=True), nn.BatchNorm2d(8),
            nn.ReflectionPad2d(1), nn.Conv2d(8, 8, kernel_size=3), nn.ReLU(inplace=True), nn.BatchNorm2d(8)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500), nn.ReLU(inplace=True),
            nn.Linear(500, 500), nn.ReLU(inplace=True),
            nn.Linear(500, 128)  # Embedding dimension
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

model = SiameseNetwork().to(device)
model.load_state_dict(torch.load("siamese_model.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

def get_embedding(image_path):
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.forward_once(image)
    return embedding

def find_best_match(input_image_path, register_path="register"):
    input_embedding = get_embedding(input_image_path)
    best_match = None
    min_dissimilarity = float("inf")
    
    for person in os.listdir(register_path):
        person_path = os.path.join(register_path, person, "register")
        if not os.path.isdir(person_path):
            continue
        
        dissimilarities = []
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            reference_embedding = get_embedding(image_path)
            dissimilarity = pairwise_distance(input_embedding, reference_embedding).item()
            dissimilarities.append(dissimilarity)
        
        avg_dissimilarity = np.mean(dissimilarities)
        if avg_dissimilarity < min_dissimilarity:
            min_dissimilarity = avg_dissimilarity
            print(f"Checking person: {person}")
            best_match = person
    
    return best_match, min_dissimilarity

app = FastAPI()

@app.post("/identify/")
async def identify_person(file: UploadFile = File(...)):
    image_path = f"temp_{file.filename}"
    with open(image_path, "wb") as buffer:
        buffer.write(await file.read())
    
    match, dissimilarity = find_best_match(image_path)
    os.remove(image_path) 

    return {"best_match": match, "dissimilarity": dissimilarity}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
