import streamlit as st
import numpy as np
import pickle
from PIL import Image, ImageOps

# 1️⃣ Load trained model
model = pickle.load(open("mnist.pkl","rb"))

st.title("🖐 Draw Digit & Predict - MNIST Model")

st.write("Draw any digit (0-9) below & click Predict")

# 2️⃣ Draw canvas
# Streamlit canvas
from streamlit_drawable_canvas import st_canvas

canvas_res = 280  # 280x280 white canvas
canvas = st_canvas(
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    height=canvas_res,
    width=canvas_res,
    drawing_mode="freedraw",
    key="canvas",
)

# 3️⃣ Predict button
if st.button("Predict Digit"):

    if canvas.image_data is not None:
        # Read drawn image
        img = Image.fromarray(canvas.image_data.astype("uint8"))

        # Convert to grayscale
        img = ImageOps.grayscale(img)

        # Resize to 28x28 for MNIST
        img = img.resize((28,28))

        # Convert to numpy and reshape
        img_arr = np.array(img).reshape(1, 784)

        # Normalize (0-255 → 0-1)
        img_arr = img_arr / 255.0

        # Predict
        result = model.predict(img_arr)[0]

        st.success(f"🎯 Predicted Digit = **{result}**")
    else:
        st.error("Draw something first!")

# 4️⃣ Show Model Score
if st.button("Show Model Accuracy"):
    from sklearn.metrics import accuracy_score

    # (Reload MNIST to check accuracy)
    from sklearn.datasets import fetch_openml
    m = fetch_openml("mnist_784", as_frame=True)
    X = m.data
    y = m.target.astype("int")

    # Take 3000 samples for fast speed
    X_test = X[:3000]
    y_test = y[:3000]

    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    st.info(f"📊 Model Accuracy = **{score*100:.2f}%**")
