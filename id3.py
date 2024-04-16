import pandas as pd
import math
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, plot_tree

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
