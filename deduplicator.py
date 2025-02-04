import pandas as pd
import streamlit as st
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import numpy as np
from pathlib import Path
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
        - unique_df: DataFrame with unique rows (default uses the first occurrence).
        - duplicates_df: DataFrame with duplicate rows and their details.
        - similarity_details: Dictionary containing similarity scores, overlap scores, and actual values for duplicates.
        - groups: List of duplicate groups (each group is a list of indices that are considered duplicates).
    """
    # Reset index to ensure unique indices
    df = df.reset_index(drop=True)
    
    # Create a preprocessed column for faster comparisons
    df['preprocessed'] = df[columns[0]].apply(preprocess_text)
    
    groups = []
    processed = set()
    
    # Identify groups of duplicates
    for i, row in df.iterrows():
        if i in processed:
            continue
        group = [i]
        for j, compare_row in df.iterrows():
            if j in processed or j == i:
                continue
            str1 = df.at[i, 'preprocessed']
            str2 = df.at[j, 'preprocessed']
            
            # Compute similarity score using fuzzy matching
            score = fuzz.ratio(str1, str2)
            
            # Also compute a "compact" score by removing all spaces
            compact_str1 = str1.replace(" ", "")
            compact_str2 = str2.replace(" ", "")
            compact_score = fuzz.ratio(compact_str1, compact_str2)
            
            # Calculate word overlap
            overlap = calculate_overlap(str1, str2)
            
            # Duplicate-capturing logic:
            if (score >= similarity_threshold and overlap >= overlap_threshold) or \
               (compact_score >= similarity_threshold) or \
               (calculate_overlap(str1, str2) == 1.0 and score >= 50):
                group.append(j)
                processed.add(j)
        if len(group) > 1:
            groups.append(group)
        processed.add(i)
    
    # Build duplicates details for reporting
    duplicates = []
    similarity_details = {}
    for group in groups:
        original_index = group[0]
        for duplicate_index in group[1:]:
            duplicates.append((duplicate_index, original_index))
            str1 = df.at[original_index, columns[0]]
            str2 = df.at[duplicate_index, columns[0]]
            score = fuzz.ratio(str1, str2)
            overlap = calculate_overlap(str1, str2)
            similarity_details[(duplicate_index, original_index)] = {
                "Scores": {columns[0]: score},
                "Overlap": overlap,
                "Actual Values": {columns[0]: {
                    "Duplicate Value": str2,
                    "Original Value": str1
                }}
            }
    
    # Create a default unique dataframe that uses the first record of each duplicate group:
    default_selected = {group[0] for group in groups}
    duplicates_all = set()
    for group in groups:
        duplicates_all.update(set(group))
    unique_indices = list(set(df.index) - (duplicates_all - default_selected))
    unique_df = df.loc[unique_indices].drop(columns=['preprocessed'])
    
    duplicates_df = df.loc[list(duplicates_all - default_selected)].drop(columns=['preprocessed'])
    # Add details for reporting (only for showing comparisons)
    duplicates_df['Duplicate Index'] = [d[0] for d in duplicates]
    duplicates_df['Original Index'] = [d[1] for d in duplicates]
    duplicates_df['Similarity Details'] = [similarity_details[d] for d in duplicates]
    
    return unique_df, duplicates_df, similarity_details, groups

# Streamlit App
def main():
    st.title("Fuzzy Duplicate Remover")
    st.write("Upload your Excel or CSV file to remove fuzzy duplicates.")

    # Use consistent keys in session state to preserve values on re-run
    if "reset" not in st.session_state:
        st.session_state.reset = False

    # File uploader with a dedicated key so the uploaded file remains available
    uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "csv"], key="uploaded_file")
    if uploaded_file is not None:
        # Process the file on first upload and store the result in session_state
        if "df" not in st.session_state:
            if uploaded_file.name.endswith(".xlsx"):
                st.session_state.df = pd.read_excel(uploaded_file, header=0)
            else:
                st.session_state.df = pd.read_csv(uploaded_file, header=0)
            
            # Convert 'Lead Originator' column to string if it exists
            if "Lead Originator" in st.session_state.df.columns:
                st.session_state.df["Lead Originator"] = st.session_state.df["Lead Originator"].astype(str)
        df = st.session_state.df

        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        # Allow user to select columns for duplicate check. Use a key to preserve selection.
        columns = st.multiselect("Select columns for duplicate check", df.columns, key="columns")
        if columns:
            similarity_threshold = st.slider("Similarity Score Threshold", 0, 100, 80, key="sim_thresh")
            overlap_threshold = st.slider("Word Overlap Threshold", 0.0, 1.0, 0.5, key="overlap_thresh")

            # When the Remove Duplicates button is clicked, process the duplicates
            if st.button("Remove Duplicates", key="remove_duplicates"):
                with st.spinner("Processing duplicates..."):
                    unique_df, duplicates_df, similarity_details, dup_groups = remove_fuzzy_duplicates(
                        df,
                        columns,
                        similarity_threshold=similarity_threshold,
                        overlap_threshold=overlap_threshold
                    )
                st.session_state.unique_df = unique_df
                st.session_state.duplicates_df = duplicates_df
                st.session_state.dup_groups = dup_groups
                st.session_state.similarity_details = similarity_details

            if "dup_groups" in st.session_state:
                st.write(f"Initial number of duplicates detected: {len(st.session_state.duplicates_df)}")
                st.write("Duplicates Detected (for review):")
                st.dataframe(st.session_state.duplicates_df)

                # Review Duplicate Groups using checkboxes.
                # Default: first instance is kept (unchecked) and the rest are flagged for removal (checked).
                if st.session_state.dup_groups:
                    st.header("Review Duplicate Groups")
                    removal_set = set()
                    for idx, group in enumerate(st.session_state.dup_groups):
                        st.write(f"**Duplicate Group {idx + 1}:**")
                        for i in group:
                            # For the original row, display as "Keep" (no similarity details)
                            if i == group[0]:
                                label = f"Keep Row {i}: {df.at[i, columns[0]]} (Original)"
                                default_checked = False
                            else:
                                details = st.session_state.similarity_details.get((i, group[0]), {})
                                score = details.get("Scores", {}).get(columns[0], "N/A")
                                overlap = details.get("Overlap", 0)
                                label = f"Remove Row {i}: {df.at[i, columns[0]]} (Similarity: {score}, Overlap: {overlap:.2f})"
                                default_checked = True
                            # Each checkbox gets a unique key so its state is preserved between re-runs
                            if st.checkbox(label, value=default_checked, key=f"group_{idx}_{i}"):
                                removal_set.add(i)

                    if st.button("Finalize Selections", key="finalize"):
                        final_df = df.drop(index=removal_set)
                        st.write("Final Unique Data:")
                        st.dataframe(final_df)

                        st.download_button(
                            label="Download Final Data as CSV",
                            data=final_df.to_csv(index=False).encode("utf-8"),
                            file_name="final_unique_data.csv",
                            mime="text/csv",
                        )
                        st.session_state.final_df = final_df

                # Also provide the default unique output using first occurrences for reference.
                st.write(f"Default Unique Data (using first occurrence of each duplicate group): {len(st.session_state.unique_df)} rows")
                st.dataframe(st.session_state.unique_df)
                st.download_button(
                    label="Download Default Unique Data as CSV",
                    data=st.session_state.unique_df.to_csv(index=False).encode("utf-8"),
                    file_name="default_unique_data.csv",
                    mime="text/csv",
                )
                st.download_button(
                    label="Download Duplicates as CSV",
                    data=st.session_state.duplicates_df.to_csv(index=False).encode("utf-8"),
                    file_name="duplicates.csv",
                    mime="text/csv",
                )
        else:
            st.warning("Please select at least one column for duplicate check.")
    else:
        st.info("Please upload a file.")

    # Reset the app (clearing session state) if needed.
    if st.button("Reset", key="reset_button"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

if __name__ == "__main__":
    main()