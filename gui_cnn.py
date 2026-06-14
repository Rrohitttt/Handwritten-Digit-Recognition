import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
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


# Load Model
model = CNN()
model.load_state_dict(
    torch.load(
        "digit_cnn.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()


class DigitRecognizer:

    def __init__(self, root):

        self.root = root
        self.root.title("CNN Digit Recognizer")

        self.canvas = tk.Canvas(
            root,
            width=280,
            height=280,
            bg="white"
        )

        self.canvas.pack(pady=10)

        self.image = Image.new(
            "L",
            (280, 280),
            color=255
        )

        self.draw = ImageDraw.Draw(
            self.image
        )

        self.canvas.bind(
            "<B1-Motion>",
            self.paint
        )

        self.predict_btn = tk.Button(
            root,
            text="Predict",
            command=self.predict_digit
        )

        self.predict_btn.pack()

        self.clear_btn = tk.Button(
            root,
            text="Clear",
            command=self.clear_canvas
        )

        self.clear_btn.pack()

        self.result = tk.Label(
            root,
            text="Draw a digit",
            font=("Arial", 18)
        )

        self.result.pack(pady=10)

    def paint(self, event):

        r = 12

        self.canvas.create_oval(
            event.x-r,
            event.y-r,
            event.x+r,
            event.y+r,
            fill="black",
            outline="black"
        )

        self.draw.ellipse(
            (
                event.x-r,
                event.y-r,
                event.x+r,
                event.y+r
            ),
            fill=0
        )

    def clear_canvas(self):

        self.canvas.delete("all")

        self.image = Image.new(
            "L",
            (280, 280),
            color=255
        )

        self.draw = ImageDraw.Draw(
            self.image
        )

        self.result.config(
            text="Draw a digit"
        )

    def predict_digit(self):

        img = self.image.copy()

        img = ImageOps.invert(img)

        img = img.resize((28, 28))

        img = np.array(img)

        img = img.astype(np.float32)

        img = (img / 255.0 - 0.5) / 0.5

        img = torch.tensor(img)

        img = img.unsqueeze(0)
        img = img.unsqueeze(0)

        with torch.no_grad():

            output = model(img)

            probs = torch.softmax(
                output,
                dim=1
            )

            confidence = torch.max(
                probs
            ).item() * 100

            prediction = torch.argmax(
                probs
            ).item()

        self.result.config(
            text=f"Digit: {prediction} ({confidence:.1f}%)"
        )


root = tk.Tk()

app = DigitRecognizer(root)

root.mainloop()