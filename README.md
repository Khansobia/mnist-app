This project is a handwritten digit recognition web app trained on the MNIST dataset.
The model was created in Jupyter Notebook, saved using pickle, and deployed using Streamlit.

Users can draw a digit on canvas → the model predicts the digit → shows accuracy & confidence score.

📂 Project Structure
📁 project-folder
│-- mnist.pkl           # Trained ML model file
│-- mnist.-app.py        # Streamlit web application
│-- README.md           # Project documentation

✅ Features

Loads MNIST trained ML model

User draws digit on screen and receives predicted output

Shows prediction score / confidence

Clean UI using Streamlit

Beginner-friendly ML deployment project

🛠️ Technologies Used
Tool / Library	Purpose
Python	Main programming language
Scikit-Learn	Training MNIST model
Pickle	Saving trained model
Streamlit	Web UI for prediction
Numpy / Pandas	Data handling
Matplotlib	Visualization during training
🚀 How to Run Project
1️⃣ Install Requirements
pip install streamlit scikit-learn numpy pandas matplotlib pillow

2️⃣ Run Streamlit App
streamlit run mnist_app.py

🧪 Model Training Summary

Dataset: MNIST (70,000 digit images)

Input Shape: 784 (28x28 pixels)

Model: Logistic Regression

Train/Test Split: 80/20

Output: Pickle file – mnist.pkl

🎨 App Preview

➡️ Draw a digit on canvas
➡️ Click Predict
➡️ Model prints predicted digit & confidence score

🎯 Learning Outcomes

Fetch and preprocess MNIST dataset

Train a classifier model in Python

Save & load model using pickle

Deploy ML model using Streamlit

End-to-end Machine Learning pipeline

👩‍💻 Author
 Sobia Khan 

Course: DCA Machine Learning Practice Project

Goal: Learn ML + Deployment

⭐ Future Improvements

📌 License

This project is open source for learning. Feel free to use and modify.
