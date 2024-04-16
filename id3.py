import pandas as pd
import numpy as np
import streamlit as st

# Define the Node class to represent a node in the decision tree
class Node:
    def __init__(self, feature=None, value=None, results=None, left=None, right=None):
        self.feature = feature  # Feature to split on
        self.value = value      # Value of the feature to make the split
        self.results = results  # Dict of results for this node, None for everything except endpoints
        self.left = left        # Left subtree
        self.right = right      # Right subtree

# Define the ID3 algorithm function
def id3(data, target_column):
    # If all data points have the same class label, return a leaf node with that label
    if len(set(data[target_column])) == 1:
        return Node(results={target_column: data[target_column].iloc[0]})
    
    # If no features are left, return a leaf node with the majority class label
    if len(data.columns) == 1:
        majority_class = data[target_column].mode()[0]
        return Node(results={target_column: majority_class})
    
    # Find the best feature to split on
    best_feature, best_value = find_best_split(data, target_column)
    
    # Split the data based on the best feature and value
    left_data = data[data[best_feature] <= best_value]
    right_data = data[data[best_feature] > best_value]
    
    # Recursively build the left and right subtrees
    left_subtree = id3(left_data, target_column)
    right_subtree = id3(right_data, target_column)
    
    # Return the current node with the best feature and value
    return Node(feature=best_feature, value=best_value, left=left_subtree, right=right_subtree)

# Function to find the best feature and value to split on
def find_best_split(data, target_column):
    best_gain = 0
    best_feature = None
    best_value = None
    
    # Calculate the entropy of the entire dataset
    entropy_total = calculate_entropy(data[target_column])
    
    # Loop over each feature
    for feature in data.columns[:-1]:  # Exclude the target column
        values = sorted(set(data[feature]))
        
        # Try all possible split points
        for i in range(len(values) - 1):
            split_value = (values[i] + values[i+1]) / 2
            
            # Split the data
            left_data = data[data[feature] <= split_value]
            right_data = data[data[feature] > split_value]
            
            # Calculate the information gain
            info_gain = entropy_total - (len(left_data) / len(data) * calculate_entropy(left_data[target_column]) +
                                         len(right_data) / len(data) * calculate_entropy(right_data[target_column]))
            
            # Update the best split if this one is better
            if info_gain > best_gain:
                best_gain = info_gain
                best_feature = feature
                best_value = split_value
    
    return best_feature, best_value

# Function to calculate entropy
def calculate_entropy(column):
    p = column.value_counts(normalize=True)
    return -(p * np.log2(p)).sum()

# Function to make predictions
def predict(node, instance):
    if node.results is not None:
        return node.results
    
    if instance[node.feature] <= node.value:
        return predict(node.left, instance)
    else:
        return predict(node.right, instance)

# Main function to create the Streamlit app
def main():
    st.title("ID3 Algorithm with Streamlit")
    
    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        st.write("### Dataset:")
        df = pd.read_csv(uploaded_file)
        st.write(df.head())

        # Select target column
        target_column = st.selectbox("Select target column", df.columns)
        
        # Build the decision tree
        root = id3(df, target_column)
        
        # Show the decision tree
        st.write("### Decision Tree:")
        st.write(print_tree(root))

        # Make predictions
        st.write("### Make Predictions:")
        example_instance = {}
        for feature in df.columns[:-1]:
            example_instance[feature] = st.number_input(f"Enter value for {feature}", step=0.1)
        prediction = predict(root, example_instance)
        st.write("Prediction:", prediction)

# Function to print the decision tree
def print_tree(node, spacing=""):
    if node.results is not None:
        return str(node.results)
    else:
        text = f"{node.feature} <= {node.value}\n{spacing}├─ Left: {print_tree(node.left, spacing + '|  ')}\n{spacing}└─ Right: {print_tree(node.right, spacing + '   ')}"
        return text

if __name__ == "__main__":
    main()
