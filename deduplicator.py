import pandas as pd
import streamlit as st
from fuzzywuzzy import fuzz
import re

# Function to preprocess text
def preprocess_text(text):
    """
    Enhanced preprocessing to handle variations in business names
    """
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    # Remove common business suffixes
    suffixes = ["limited", "ltd", "inc", "corporation", "llc", "fzc", "fzco", "international", "intl"]
    text = " ".join([word for word in text.split() if word not in suffixes])
    
    # Remove special characters but keep spaces
    text = re.sub(r"[^\w\s]", "", text)
    
    # Handle empty strings after preprocessing
    if not text.strip():
        return ""
    
    # Normalize spaces and join words
    return " ".join(text.split())

def calculate_overlap(str1, str2):
    """
    Enhanced overlap calculation with better handling of variations
    """
    words1 = set(str1.lower().split())
    words2 = set(str2.lower().split())
    
    # Special handling for single-word matches
    if len(words1) == 1 or len(words2) == 1:
        # If one is a single word, check if it's a subset of the other
        if words1.issubset(words2) or words2.issubset(words1):
            return 1.0
    
    # Normal overlap calculation
    overlap = len(words1 & words2) / max(len(words1), len(words2))
    return overlap

# Function to perform fuzzy duplicate removal
def remove_fuzzy_duplicates(df, columns, similarity_threshold=80, overlap_threshold=0.5):
    """
    Remove fuzzy duplicates from a DataFrame based on selected columns.
    Returns:
        - unique_df: DataFrame with unique rows.
        - duplicates_df: DataFrame with duplicate rows and their details.
        - similarity_details: Dictionary containing similarity scores and actual values for duplicates.
    """
    # Reset index to ensure unique indices
    df = df.reset_index(drop=True)
    
    # Create a preprocessed column for faster comparisons
    df['preprocessed'] = df[columns[0]].apply(preprocess_text)
    
    # Group similar values together
    groups = []
    processed = set()
    
    for i, row in df.iterrows():
        if i in processed:
            continue
        
        # Find all similar rows
        group = [i]
        for j, compare_row in df.iterrows():
            if j in processed or j == i:
                continue
            
            str1 = df.at[i, 'preprocessed']
            str2 = df.at[j, 'preprocessed']
            
            score = fuzz.ratio(str1, str2)
            overlap = calculate_overlap(str1, str2)
            
            if (score >= similarity_threshold and overlap >= overlap_threshold) or \
               (calculate_overlap(str1, str2) == 1.0 and score >= 50):
                group.append(j)
                processed.add(j)
        
        if len(group) > 1:
            groups.append(group)
        processed.add(i)
    
    # Create duplicates DataFrame
    duplicates = []
    similarity_details = {}
    
    for group in groups:
        original_index = group[0]
        for duplicate_index in group[1:]:
            duplicates.append((duplicate_index, original_index))
            # Store similarity details
            str1 = df.at[original_index, columns[0]]
            str2 = df.at[duplicate_index, columns[0]]
            score = fuzz.ratio(str1, str2)
            overlap = calculate_overlap(str1, str2)
            similarity_details[(duplicate_index, original_index)] = {
                "Scores": {columns[0]: score},
                "Actual Values": {columns[0]: {
                    "Duplicate Value": str2,
                    "Original Value": str1
                }}
            }
    
    # Create unique and duplicates DataFrames
    unique_indices = list(set(df.index) - {d[0] for d in duplicates})
    unique_df = df.loc[unique_indices].drop(columns=['preprocessed'])
    
    duplicates_indices = list({d[0] for d in duplicates})
    duplicates_df = df.loc[duplicates_indices].drop(columns=['preprocessed'])
    
    # Add duplicate details to the duplicates DataFrame
    duplicates_df['Duplicate Index'] = [d[0] for d in duplicates]
    duplicates_df['Original Index'] = [d[1] for d in duplicates]
    duplicates_df['Similarity Details'] = [similarity_details[d] for d in duplicates]
    
    return unique_df, duplicates_df, similarity_details

# Streamlit App
def main():
    st.title("Fuzzy Duplicate Remover")
    st.write("Upload your Excel or CSV file to remove fuzzy duplicates.")

    # Initialize session state for reset functionality
    if "reset" not in st.session_state:
        st.session_state.reset = False

    # Step 1: Upload file
    uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "csv"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file, header=0)
            else:
                df = pd.read_csv(uploaded_file, header=0)

            st.write("Preview of uploaded data:")
            st.dataframe(df.head())

            # Step 2: Select columns for duplicate check
            columns = st.multiselect("Select columns for duplicate check", df.columns)
            if columns:
                similarity_threshold = st.slider("Similarity Score Threshold", 0, 100, 80)
                overlap_threshold = st.slider("Word Overlap Threshold", 0.0, 1.0, 0.5)

                if st.button("Remove Duplicates"):
                    with st.spinner("Processing duplicates..."):
                        unique_df, duplicates_df, similarity_details = remove_fuzzy_duplicates(
                            df,
                            columns,
                            similarity_threshold=similarity_threshold,
                            overlap_threshold=overlap_threshold
                        )

                    # Step 4: Show changes
                    st.write(f"Number of duplicates removed: {len(duplicates_df)}")
                    st.write("Duplicates Removed:")
                    st.dataframe(duplicates_df)

                    # Display similarity details for duplicates
                    st.write("Similarity Details for Duplicates:")
                    for _, row in duplicates_df.iterrows():
                        duplicate_index = row["Duplicate Index"]
                        original_index = row["Original Index"]
                        details = row["Similarity Details"]
                        st.write(f"Duplicate Row {duplicate_index} vs Original Row {original_index}:")
                        st.json(details)  # Display scores and actual values in JSON format

                    st.write(f"Number of unique rows retained: {len(unique_df)}")
                    st.write("Unique Data:")
                    st.dataframe(unique_df)

                    # Step 5: Download the final table
                    st.download_button(
                        label="Download Unique Data as CSV",
                        data=unique_df.to_csv(index=False).encode("utf-8"),
                        file_name="unique_data.csv",
                        mime="text/csv",
                    )

                    # Option to download duplicates
                    st.download_button(
                        label="Download Duplicates as CSV",
                        data=duplicates_df.to_csv(index=False).encode("utf-8"),
                        file_name="duplicates.csv",
                        mime="text/csv",
                    )

                # Reset button
                if st.button("Reset"):
                    st.session_state.reset = True
                    st.experimental_rerun()

            else:
                st.warning("Please select at least one column for duplicate check.")

        except Exception as e:
            st.error(f"An error occurred: {e}. Please check the file format and content.")

    # Handle reset
    if st.session_state.reset:
        st.session_state.reset = False
        st.experimental_rerun()

if __name__ == "__main__":
    main()