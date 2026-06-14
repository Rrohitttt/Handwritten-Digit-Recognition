import tkinter as tk
from tkinter import Label, Button, Canvas
from PIL import Image, ImageDraw
import numpy as np
import joblib

# Load trained model
model = joblib.load("digit_model.pkl")

class DigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")

        self.canvas = Canvas(root, width=200, height=200, bg="white")
        self.canvas.pack()

        self.image = Image.new("L", (200, 200), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

        self.predict_btn = Button(
            root,
            text="Predict",
            command=self.predict_digit
        )
        self.predict_btn.pack()

        self.clear_btn = Button(
            root,
            text="Clear",
            command=self.clear_canvas
        )
        self.clear_btn.pack()

        self.result = Label(root, text="Draw a digit")
        self.result.pack()

    def paint(self, event):
        x1, y1 = event.x - 8, event.y - 8
        x2, y2 = event.x + 8, event.y + 8

        self.canvas.create_oval(
            x1, y1, x2, y2,
            fill="black",
            outline="black"
        )

        self.draw.ellipse(
            [x1, y1, x2, y2],
            fill=0
        )

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (200, 200), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.result.config(text="Draw a digit")

    def predict_digit(self):
        img = self.image.resize((8, 8))
        img = np.array(img)

        img = 16 - (img / 16)
        img = img.flatten()

        prediction = model.predict([img])[0]

        self.result.config(
            text=f"Predicted Digit: {prediction}"
        )

root = tk.Tk()
app = DigitRecognizer(root)
root.mainloop()