
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import json
import sys

# Set default encoding to utf-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Load your trained model
model = tf.keras.models.load_model('food_classification_model.keras', compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the class indices
with open('class_indices.json', 'r', encoding='utf-8') as f:
    class_indices = json.load(f)

# Reverse the class indices dictionary to map indices to class labels
class_indices = {v: k for k, v in class_indices.items()}

# Function to load and preprocess an image
def load_and_preprocess_image(img_file):
    try:
        img = keras_image.load_img(img_file, target_size=(150, 150))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

# Function to predict the class of an image
def predict_image_class(model, img_array, class_indices):
    if img_array is None:
        return "Unknown"
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_indices.get(predicted_class_index, "Unknown")
    return predicted_class_label

# Calorie map for each class
calorie_map = {
    'samosa': 250,
    'pizza': 300,
    'Burger': 500,
    'idli':150,
    'kulfi':206
    
}

# Function to calculate recommended calories
def get_recommended_calories(age, gender, height, weight, activity_level, goal):
    if gender.lower() == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    activity_factors = {
        'sedentary': 1.2,
        'lightly active': 1.375,
        'moderately active': 1.55,
        'very active': 1.725,
        'super active': 1.9
    }

    tdee = bmr * activity_factors[activity_level.lower()]

    if goal.lower() == 'lose weight':
        recommended_calories = tdee - 500
    elif goal.lower() == 'gain weight':
        recommended_calories = tdee + 500
    else:
        recommended_calories = tdee

    return recommended_calories

# Function to calculate calorie difference
def calculate_calorie_difference(recommended_calories, consumed_calories):
    return recommended_calories - consumed_calories

# Function to provide dietary recommendation
def provide_dietary_recommendation(calorie_difference):
    if calorie_difference > 0:
        return "You have remaining calories for the day. Consider consuming a balanced meal."
    elif calorie_difference < 0:
        return "You have exceeded your calorie intake for the day. Consider light snacks or low-calorie foods."
    else:
        return "You have met your calorie intake for the day. Maintain a balanced diet."

# Main Streamlit app
def main():
    st.title("Food Classifier and Calorie Counter")

    # Sidebar for user information
    st.sidebar.title("User Information")
    name = st.sidebar.text_input("Name", "John Doe")
    age = st.sidebar.number_input("Age", min_value=0, max_value=150, value=30)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    height = st.sidebar.number_input("Height (cm)", min_value=0, max_value=300, value=175)
    weight = st.sidebar.number_input("Weight (kg)", min_value=0, max_value=500, value=70)
    activity_level = st.sidebar.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Super Active"])
    goal = st.sidebar.selectbox("Goal", ["Lose Weight", "Maintain Weight", "Gain Weight"])

    # File uploader for image
    st.title("Upload an image of the dish")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Perform classification and calorie counting
        img_array = load_and_preprocess_image(uploaded_file)
        predicted_class = predict_image_class(model, img_array, class_indices)
        consumed_calories = calorie_map.get(predicted_class, 0)
        recommended_calories = get_recommended_calories(age, gender, height, weight, activity_level, goal)
        calorie_difference = calculate_calorie_difference(recommended_calories, consumed_calories)
        dietary_recommendation = provide_dietary_recommendation(calorie_difference)

        # Display results
        st.title("Result")
        st.write(f'The image is classified as: {predicted_class}')
        st.write(f'Calories in the dish: {consumed_calories}')
        st.write(f'Recommended daily calorie intake: {recommended_calories}')
        st.write(f'Calorie difference: {calorie_difference}')
        st.write(f'Dietary recommendation: {dietary_recommendation}')

if __name__ == "__main__":
    main()
