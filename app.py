import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn as nn

# CNN Architecture

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
```
    self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

    self.pool = nn.MaxPool2d(2, 2)

    self.fc1 = nn.Linear(64 * 7 * 7, 128)
    self.fc2 = nn.Linear(128, 10)

    self.relu = nn.ReLU()

def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.pool(x)

    x = self.relu(self.conv2(x))
    x = self.pool(x)

    x = x.view(-1, 64 * 7 * 7)

    x = self.relu(self.fc1(x))
    x = self.fc2(x)

    return x
```

@st.cache_resource
def load_model():
model = CNN()
model.load_state_dict(
torch.load(
"digit_cnn.pth",
map_location=torch.device("cpu")
)
)
model.eval()
return model

model = load_model()

st.title("Handwritten Digit Recognition")

canvas_result = st_canvas(
fill_color="white",
stroke_width=15,
stroke_color="white",
background_color="black",
height=280,
width=280,
drawing_mode="freedraw",
key="canvas",
)

if st.button("Predict"):

```
if canvas_result.image_data is not None:

    img = Image.fromarray(
        (canvas_result.image_data[:, :, 0]).astype("uint8")
    )

    img = img.resize((28, 28))

    img = np.array(img).astype(np.float32)

    img = (img / 255.0 - 0.5) / 0.5

    img = torch.tensor(img)

    img = img.unsqueeze(0)
    img = img.unsqueeze(0)

    with torch.no_grad():

        output = model(img)

        probs = torch.softmax(output, dim=1)

        confidence = torch.max(probs).item() * 100

        prediction = torch.argmax(probs).item()

    st.success(
        f"Predicted Digit: {prediction} ({confidence:.1f}%)"
    )
```
