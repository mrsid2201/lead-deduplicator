import pandas as pd
import streamlit as st
from fuzzywuzzy import fuzz
import re

# Function to preprocess text
def preprocess_text(text):
    """
    Normalize text by converting to lowercase and stripping punctuation, but retain spaces.
    """
    if pd.isna(text):  # Handle NaN values
        return ""
    original_text = str(text)  # Store original text for debugging
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation but retain spaces
    print(f"Original: {original_text} -> Preprocessed: {text}")  # Debug statement
    return text

# Function to perform fuzzy duplicate removal
def remove_fuzzy_duplicates(df, columns, threshold=80):
    """
    Remove fuzzy duplicates from a DataFrame based on selected columns.
    Rows are considered duplicates if any one of the selected columns meets the threshold.
    For the column 'Main Phone #', only exact matches are considered.
    Returns:
        - unique_df: DataFrame with unique rows.
        - duplicates_df: DataFrame with duplicate rows and their details.
        - similarity_details: Dictionary containing similarity scores and actual values for duplicates.
    """
    duplicates = []
    unique_data = []
    similarity_details = {}  # Store similarity scores and actual values for duplicates
    total_rows = len(df)
    progress_bar = st.progress(0)  # Initialize progress bar

    for index, row in df.iterrows():
        match_found = False
        for unique_row in unique_data:
            # Check if any one of the selected columns meets the threshold
            is_duplicate = False
            scores = {}  # Store scores for the current comparison
            actual_values = {}  # Store actual values for the current comparison
            for col in columns:
                if col == "Main Phone #":
                    # Perform exact match for 'Main Phone #'
                    str1 = str(row[col]).strip()  # Exact match, no preprocessing
                    str2 = str(unique_row[col]).strip()
                    if str1 == str2:
                        is_duplicate = True
                        scores[col] = 100  # Exact match score is 100
                        actual_values[col] = {
                            "Duplicate Value": str1,
                            "Original Value": str2,
                        }
                        break
                else:
                    # Perform fuzzy match for other columns
                    str1 = preprocess_text(row[col])
                    str2 = preprocess_text(unique_row[col])
                    score = fuzz.token_set_ratio(str1, str2)  # Use fuzz.token_set_ratio for better matching
                    scores[col] = score  # Store the score for this column
                    actual_values[col] = {
                        "Duplicate Value": row[col],
                        "Original Value": unique_row[col],
                    }
                    print(f"Comparing '{str1}' (Row {index}) with '{str2}' (Row {unique_row.name}) in column '{col}': Score = {score}")  # Debug statement
                    if score >= threshold:
                        is_duplicate = True
                        break  # No need to check other columns if one meets the threshold

            if is_duplicate:
                duplicates.append((index, unique_row.name))
                similarity_details[(index, unique_row.name)] = {
                    "Scores": scores,
                    "Actual Values": actual_values,
                }  # Store scores and actual values for this duplicate pair
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

    # Add similarity details to the duplicates DataFrame
    duplicates_df["Similarity Details"] = duplicates_df.apply(
        lambda row: similarity_details.get((row["Duplicate Index"], row["Original Index"]), {}), axis=1
    )

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