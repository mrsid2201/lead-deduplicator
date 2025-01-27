import pandas as pd
import streamlit as st
from fuzzywuzzy import fuzz
import re

# Function to preprocess text
def preprocess_text(text):
    """
    Normalize text by removing extra spaces, converting to lowercase, and stripping punctuation.
    """
    if pd.isna(text):  # Handle NaN values
        return ""
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r"\s+", "", text)  # Remove all spaces
    text = re.sub(r"[^\w]", "", text)  # Remove punctuation
    print(f"Original: {text} -> Preprocessed: {text}")  # Debug statement
    return text

# Function to perform fuzzy duplicate removal
def remove_fuzzy_duplicates(df, columns, threshold=80):
    """
    Remove fuzzy duplicates from a DataFrame based on selected columns.
    Rows are considered duplicates only if all selected columns meet the threshold.
    """
    duplicates = []
    unique_data = []

    for index, row in df.iterrows():
        match_found = False
        for unique_row in unique_data:
            # Check if all selected columns meet the threshold
            is_duplicate = True
            for col in columns:
                str1 = preprocess_text(row[col])
                str2 = preprocess_text(unique_row[col])
                score = fuzz.token_sort_ratio(str1, str2)
                print(f"Comparing '{str1}' (Row {index}) with '{str2}' (Row {unique_row.name}) in column '{col}': Score = {score}")  # Debug statement
                if score < threshold:
                    is_duplicate = False
                    break
            if is_duplicate:
                duplicates.append((index, unique_row.name))
                match_found = True
                break
        if not match_found:
            unique_data.append(row)

    # Create a DataFrame of unique rows
    unique_df = pd.DataFrame(unique_data).reset_index(drop=True)

    # Create a DataFrame of duplicates with details
    duplicates_df = pd.DataFrame(duplicates, columns=["Duplicate Index", "Original Index"])
    return unique_df, duplicates_df

# Streamlit App
def main():
    st.title("Fuzzy Duplicate Remover")
    st.write("Upload your Excel or CSV file to remove fuzzy duplicates.")

    # Step 1: Upload file
    uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "csv"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, header=0)
        else:
            df = pd.read_csv(uploaded_file, header=0)

        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        # Step 2: Select columns for duplicate check
        columns = st.multiselect("Select columns for duplicate check", df.columns)
        if columns:
            threshold = st.slider("Set fuzzy match threshold (0-100)", 0, 100, 80)

            if st.button("Remove Duplicates"):
                # Step 3: Perform fuzzy duplicate removal
                unique_df, duplicates_df = remove_fuzzy_duplicates(df, columns, threshold)

                # Step 4: Show changes
                st.write("Duplicates Removed:")
                st.dataframe(duplicates_df)

                st.write("Unique Data:")
                st.dataframe(unique_df)

                # Step 5: Download the final table
                st.download_button(
                    label="Download Unique Data as CSV",
                    data=unique_df.to_csv(index=False).encode("utf-8"),
                    file_name="unique_data.csv",
                    mime="text/csv",
                )

if __name__ == "__main__":
    main()