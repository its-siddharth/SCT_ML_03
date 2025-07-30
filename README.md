🐾 Pet Classifier (Cat vs Dog)
This is a simple AI-based image classifier built with Streamlit that predicts whether an uploaded image is of a Cat 🐱 or a Dog 🐶.

🚀 Features
Upload any image of a cat or dog

Instantly predicts whether it’s a cat or a dog

Displays confidence score

Clean, minimal interface

🛠 Tech Stack
Python

Streamlit (Web UI)

OpenCV (Image processing)

NumPy (Numerical operations)

Pickle (Loading trained ML model)

📦 Installation
1️⃣ Clone the repository

bash
Copy
Edit
git clone https://github.com/yourusername/pet-classifier.git
cd pet-classifier
2️⃣ Create & activate a virtual environment (optional but recommended)

bash
Copy
Edit
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On macOS/Linux
3️⃣ Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
▶️ Usage
1️⃣ Run the Streamlit app

bash
Copy
Edit
streamlit run app.py
2️⃣ Upload your image using the file uploader in the web interface.

3️⃣ See the prediction (Cat 🐱 or Dog 🐶) along with the confidence score.

📂 Project Structure
sql
Copy
Edit
pet-classifier/
│-- app.py              # Main Streamlit app
│-- model.pkl           # Pre-trained model
│-- requirements.txt    # Python dependencies
│-- README.md           # Project documentation
📌 Notes
Ensure your image is clear and well-lit for better accuracy.

The model expects images to be 64×64 pixels internally.

Preprocessing ensures exactly 12288 features before prediction.

📜 License
This project is open-source and free to use for educational purposes.

