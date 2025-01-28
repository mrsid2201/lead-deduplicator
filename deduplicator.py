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
    original_text = str(text)  # Store original text for debugging
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r"\s+", "", text)  # Remove all spaces
    text = re.sub(r"[^\w]", "", text)  # Remove punctuation
    print(f"Original: {original_text} -> Preprocessed: {text}")  # Debug statement
    return text

# Function to perform fuzzy duplicate removal
def remove_fuzzy_duplicates(df, columns, threshold=80):
    """
    Remove fuzzy duplicates from a DataFrame based on selected columns.
    Rows are considered duplicates if any one of the selected columns meets the threshold.
    """
    duplicates = []
    unique_data = []
    total_rows = len(df)
    progress_bar = st.progress(0)  # Initialize progress bar

    for index, row in df.iterrows():
        match_found = False
        for unique_row in unique_data:
            # Check if any one of the selected columns meets the threshold
            is_duplicate = False
            for col in columns:
                str1 = preprocess_text(row[col])
                str2 = preprocess_text(unique_row[col])
                score = fuzz.token_sort_ratio(str1, str2)
                print(f"Comparing '{str1}' (Row {index}) with '{str2}' (Row {unique_row.name}) in column '{col}': Score = {score}")  # Debug statement
                if score >= threshold:
                    is_duplicate = True
                    break  # No need to check other columns if one meets the threshold

            if is_duplicate:
                duplicates.append((index, unique_row.name))
                match_found = True
                break  # No need to check other unique rows if a match is found

        if not match_found:
            unique_data.append(row)

        # Update progress bar
        progress = (index + 1) / total_rows
        progress_bar.progress(progress)

    # Create a DataFrame of unique rows
    unique_df = pd.DataFrame(unique_data).reset_index(drop=True)

    # Create a DataFrame of duplicates with details
    duplicates_df = pd.DataFrame(duplicates, columns=["Duplicate Index", "Original Index"])
    return unique_df, duplicates_df

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
                threshold = st.slider("Set fuzzy match threshold (0-100)", 0, 100, 80)

                if st.button("Remove Duplicates"):
                    # Step 3: Perform fuzzy duplicate removal
                    unique_df, duplicates_df = remove_fuzzy_duplicates(df, columns, threshold)

                    # Step 4: Show changes
                    st.write(f"Number of duplicates removed: {len(duplicates_df)}")
                    st.write("Duplicates Removed:")
                    st.dataframe(duplicates_df)

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