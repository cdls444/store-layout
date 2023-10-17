# Import the necessary libraries and modules
import streamlit as st
import pandas as pd
import pickle
import random

# Load the Random Forest model
with open('./models/model_rfc.pkl', 'rb') as f:
    loaded_model_rfc = pickle.load(f)

# Streamlit UI components
st.title("Shelf Layer Recommendation App")

# Input fields for the user
item_width_cm = st.number_input("Item Width (cm)", value=200)
item_height_cm = st.number_input("Item Height (cm)", value=250)
item_wide_cm = st.number_input("Item Wide (cm)", value=180)
item_weight_gram = st.number_input("Item Weight (grams)", value=180)
item_quantity = st.number_input("Quantity of Items", value=300)

# Predict the recommended shelf layer and calculate distribution
if st.button("Get Recommendation"):
    # Create a user input data frame with matching feature names and order
    user_input = pd.DataFrame({
        'item_width_cm': [item_width_cm],
        'item_height_cm': [item_height_cm],
        'item_wide_cm': [item_wide_cm],
        'item_weight_gram': [item_weight_gram]
    })

    # Ensure that the columns are in the same order as used during training
    user_input = user_input[['item_width_cm', 'item_height_cm', 'item_wide_cm', 'item_weight_gram']]

    # Predict the recommended shelf layer
    recommended_layer = loaded_model_rfc.predict(user_input)

    # Predict the probabilities for each class
    predicted_probabilities = loaded_model_rfc.predict_proba(user_input)

    # Create a DataFrame with class probabilities
    probability_df = pd.DataFrame(predicted_probabilities, columns=loaded_model_rfc.classes_)

    # Get the top 4 recommendations based on probabilities
    top_4_recommendations = probability_df.iloc[0].nlargest(4)

    # Calculate distribution
    total_quantity = item_quantity
    distribution = {}
    for layer in loaded_model_rfc.classes_:
        probability = probability_df[layer].values[0]
        quantity_on_layer = int((probability * item_quantity) + 0.5)  # Round to the nearest integer
        distribution[layer] = quantity_on_layer
        total_quantity -= quantity_on_layer

    # Distribute any remaining items
    distribution[recommended_layer[0]] += total_quantity

    # Create a DataFrame for the recommendations and distribution (Layout 1)
    recommendations_and_distribution_layout1 = pd.DataFrame({
        'Shelf Layer': list(distribution.keys()),
        'Percentage': [f"{(distribution[layer] / item_quantity) * 100:.2f}%" for layer in distribution.keys()],
        'Quantity': list(distribution.values())
    })

    st.write(f"Recommended Shelf Layer: {recommended_layer[0]}")

    # Display Layout 1
    st.write("Layout 1:")
    st.table(recommendations_and_distribution_layout1)

    # Shuffle distribution for Layout 2 (random values)
    shuffled_distribution_layout2 = distribution.copy()
    for layer in shuffled_distribution_layout2:
        shuffled_distribution_layout2[layer] = random.randint(0, item_quantity)

    recommendations_and_distribution_layout2 = pd.DataFrame({
        'Shelf Layer': list(shuffled_distribution_layout2.keys()),
        'Percentage': [f"{(shuffled_distribution_layout2[layer] / item_quantity) * 100:.2f}%" for layer in shuffled_distribution_layout2.keys()],
        'Quantity': list(shuffled_distribution_layout2.values())
    })
    st.write("Layout 2 (Random Distribution):")
    st.table(recommendations_and_distribution_layout2)

    # Shuffle distribution for Layout 3 (random values)
    shuffled_distribution_layout3 = distribution.copy()
    for layer in shuffled_distribution_layout3:
        shuffled_distribution_layout3[layer] = random.randint(0, item_quantity)

    recommendations_and_distribution_layout3 = pd.DataFrame({
        'Shelf Layer': list(shuffled_distribution_layout3.keys()),
        'Percentage': [f"{(shuffled_distribution_layout3[layer] / item_quantity) * 100:.2f}%" for layer in shuffled_distribution_layout3.keys()],
        'Quantity': list(shuffled_distribution_layout3.values())
    })
    st.write("Layout 3 (Random Distribution):")
    st.table(recommendations_and_distribution_layout3)

    # Shuffle distribution for Layout 4 (random values)
    shuffled_distribution_layout4 = distribution.copy()
    for layer in shuffled_distribution_layout4:
        shuffled_distribution_layout4[layer] = random.randint(0, item_quantity)

    recommendations_and_distribution_layout4 = pd.DataFrame({
        'Shelf Layer': list(shuffled_distribution_layout4.keys()),
        'Percentage': [f"{(shuffled_distribution_layout4[layer] / item_quantity) * 100:.2f}%" for layer in shuffled_distribution_layout4.keys()],
        'Quantity': list(shuffled_distribution_layout4.values())
    })
    st.write("Layout 4 (Random Distribution):")
    st.table(recommendations_and_distribution_layout4)