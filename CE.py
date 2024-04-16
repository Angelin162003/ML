import streamlit as st
import numpy as np

def learn_candidate_elimination(concepts, target):
    specific_h = concepts[0].copy()
    general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]

    for i, h in enumerate(concepts):
        if target[i] == "yes":
            specific_h = [h_i if h_i == specific_h_i else '?' for h_i, specific_h_i in zip(h, specific_h)]
            general_h = [[h_i if h_i == specific_h_i or g_i == '?' else g_i
                          for h_i, specific_h_i, g_i in zip(h, specific_h, g)] for g in general_h]
        elif target[i] == "no":
            general_h = [[specific_h_i if h_i != specific_h_i and g_i == '?' else g_i
                          for h_i, specific_h_i, g_i in zip(h, specific_h, g)] for g in general_h]

    # Remove any general hypotheses that have all attributes as '?'
    general_h = [g for g in general_h if not all(attr == '?' for attr in g)]

    return specific_h, general_h

def main():
    st.title("Candidate Elimination Algorithm")

    # Example data
    concepts = np.array([
        ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'],
        ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change']
    ])
    target = np.array(['yes', 'yes', 'no', 'yes'])

    s_final, g_final = learn_candidate_elimination(concepts, target)

    st.write("Final Specific Hypothesis:")
    st.write(s_final)

    st.write("Final General Hypotheses:")
    for hypothesis in g_final:
        st.write(hypothesis)

if __name__ == "__main__":
    main() ex 01
[16/04, 10:02 am] Divya: import pandas as pd
import math
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Function to calculate entropy
def calculate_entropy(data, target_column):
    try:
        target_values = data[target_column].unique()
    except KeyError:
        st.error(f"Column '{target_column}' not found in the dataset.")
        return None

    total_rows = len(data)
    entropy = 0

    for value in target_values:
        value_count = len(data[data[target_column] == value])
        proportion = value_count / total_rows
        entropy -= proportion * math.log2(proportion)

    return entropy

# Main function
def main():
    st.title("Decision Tree Classifier with Streamlit")

    # Read the dataset
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.write("### Dataset:")
        st.write(df.head())

        # Allow user to select target column
        target_column = st.selectbox("Select the target column", df.columns)

        # Check if selected target column exists
        if target_column not in df.columns:
            st.error(f"Selected target column '{target_column}' not found in the dataset.")
            return

        # Calculate entropy of the dataset
        entropy_outcome = calculate_entropy(df, target_column)
        if entropy_outcome is not None:
            st.write(f"Entropy of the dataset: {entropy_outcome}")

            # Feature selection for the first step in making the decision tree
            selected_feature = st.selectbox("Select feature for the decision tree", df.columns)

            # Create a decision tree
            clf = DecisionTreeClassifier(criterion='entropy', max_depth=1)
            X = df[[selected_feature]]
            y = df[target_column]  # Use user-selected target column
            clf.fit(X, y)

            # Plot the decision tree
            try:
                class_names = [str(c) for c in sorted(y.unique())]
                plt.figure(figsize=(8, 6))
                plot_tree(clf, feature_names=[selected_feature], class_names=class_names, filled=True, rounded=True)
                st.pyplot()
            except IndexError as e:
                st.error("Error occurred while plotting the decision tree:", e)

if __name__ == "__main__":
    main()
