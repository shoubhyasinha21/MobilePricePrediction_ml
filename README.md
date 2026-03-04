📱 Mobile Price Prediction App
This is a Machine Learning web application built with Streamlit that predicts the price category of a mobile phone based on its technical specifications. The model classifies phones into four categories: Low Price, Medium Price, High Price, and Very High Price.

🚀 Features
Real-time Prediction: Enter specifications and get instant results.

Interactive UI: User-friendly interface built with Streamlit.

Scalable Architecture: Uses a pre-trained RandomForest model and StandardScaler for accurate predictions.

🛠️ Tech Stack
Frontend: Streamlit

Machine Learning: Scikit-learn

Data Processing: Pandas, NumPy

Model Serialization: Joblib

📁 Project Structure
💻 Installation & Setup
1. Clone the repository
2. Set up Virtual Environment
3. Install Dependencies
4. Train the Model
Ensure mobile_dataset.csv is in the root directory, then run:

5. Run the App
📊 Dataset Features
The model is trained on the following key features:

Battery Power: Total energy a battery can store in one time (mAh).

RAM: Random Access Memory in MegaBytes.

Pixel Resolution: Height and Width in pixels.

Mobile Weight: Weight of the mobile phone in grams.

Internal Memory: Capacity in GigaBytes.
