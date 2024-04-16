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
    main()
