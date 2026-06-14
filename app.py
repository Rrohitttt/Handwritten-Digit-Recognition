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


@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(
        torch.load(
            "digit_cnn.pth",
            map_location=torch.device("cpu"),
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
    if canvas_result.image_data is not None:
        # Convert to grayscale
        pil_img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")

        def preprocess(pil_img):
            arr = np.array(pil_img)
            # Find bounding box of the drawn digit (non-zero pixels)
            non_empty_rows = np.where(np.any(arr > 0, axis=1))[0]
            non_empty_cols = np.where(np.any(arr > 0, axis=0))[0]

            if non_empty_rows.size and non_empty_cols.size:
                top, bottom = non_empty_rows[0], non_empty_rows[-1]
                left, right = non_empty_cols[0], non_empty_cols[-1]
                crop = pil_img.crop((left, top, right + 1, bottom + 1))
            else:
                crop = pil_img

            # Resize preserving aspect ratio to fit in 20x20 box
            max_side = 20
            w, h = crop.size
            if w > h:
                new_w = max_side
                new_h = int(round((max_side * h) / w))
            else:
                new_h = max_side
                new_w = int(round((max_side * w) / h))

            if new_w == 0:
                new_w = 1
            if new_h == 0:
                new_h = 1

            # Use LANCZOS for high-quality resize; ANTIALIAS is deprecated in newer Pillow versions
            resized = crop.resize((new_w, new_h), Image.LANCZOS)

            # Paste into 28x28 image and center
            new_img = Image.new('L', (28, 28), 0)
            paste_x = (28 - new_w) // 2
            paste_y = (28 - new_h) // 2
            new_img.paste(resized, (paste_x, paste_y))

            return new_img

        proc = preprocess(pil_img)

        # Show processed 28x28 image for debugging
        st.image(proc.resize((140, 140)), caption="Processed 28x28 input (upsampled)")

        img = np.array(proc).astype(np.float32)
        img = img / 255.0
        img = (img - 0.5) / 0.5

        img = torch.tensor(img)
        img = img.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            probs = torch.softmax(output, dim=1)
            confidence = torch.max(probs).item() * 100
            prediction = torch.argmax(probs).item()

        st.success(f"Predicted Digit: {prediction} ({confidence:.1f}%)")
    else:
        st.warning("Draw a digit on the canvas before predicting.")
