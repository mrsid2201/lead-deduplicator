import pandas as pd
import streamlit as st
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import numpy as np
from pathlib import Path
import re
import time
import logging
import json
from datetime import datetime
import os

# Set up logging configuration - terminal logs only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration and constants
DEFAULT_SIMILARITY_THRESHOLD = 80
DEFAULT_OVERLAP_THRESHOLD = 0.5

# Configurable business name patterns
BUSINESS_PATTERNS = {
    "suffixes": [
        "limited", "ltd", "inc", "incorporation", "corporation", "corp",
        "llc", "fzc", "fzco", "international", "intl", "middle east",
        "trading", "holdings", "group", "company", "co", "est", "establishment",
        "enterprises", "ventures", "solutions", "services", "industries"
    ],
    "legal_forms": [
        "llc", "fzc", "fzco", "ltd", "inc", "corp", "co"
    ],
    "business_types": [
        "trading", "holdings", "group", "enterprises", "ventures",
        "solutions", "services", "industries", "establishment"
    ]
}

# Configurable address patterns
ADDRESS_PATTERNS = {
    "unit_types": [
        ("st", "street"),
        ("bldg", "building"),
        ("ofc", "office"),
        ("flr", "floor"),
        ("apt", "apartment"),
        ("po box", "po box"),
        ("pobox", "po box"),
        ("ste", "suite"),
        ("unit", "unit"),
        ("rm", "room")
    ],
    "location_types": [
        ("dhcc", "dubai healthcare city"),
        ("difc", "dubai international financial center"),
        ("dafza", "dubai airport free zone"),
        ("jafza", "jebel ali free zone")
    ]
}

# CSS styles for the UI
def load_css():
    """Load CSS styles for the application"""
    st.markdown("""
    <style>
    .duplicate-header {
        font-weight: bold;
        color: #2C3E50;
        font-size: 1.1rem;
        margin-bottom: 10px;
    }
    .score-badge {
        display: inline-block;
        padding: 2px 6px;
        border-radius: 10px;
        font-size: 0.8rem;
        margin-left: 5px;
        color: white;
        font-weight: normal;
    }
    .high-score {
        background-color: #27AE60;
    }
    .medium-score {
        background-color: #F39C12;
    }
    .low-score {
        background-color: #E74C3C;
    }
    .groups-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(100%, 1fr));
        gap: 15px;
        margin-bottom: 20px;
        width: 100%;
    }
    .group-card {
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        width: 100%;
        box-sizing: border-box;
    }
    .group-title {
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 15px;
        padding-bottom: 8px;
        border-bottom: 1px solid #E0E0E0;
        color: #2C3E50;
    }
    /* Responsive table styles */
    .stDataFrame {
        width: 100%;
        overflow-x: auto;
    }
    /* Checkbox label styles */
    .stCheckbox label {
        word-wrap: break-word;
        max-width: 100%;
        display: inline-block;
    }
    /* Make buttons more visible */
    .stButton > button {
        background-color: #3498DB;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
    }
    .stButton > button:hover {
        background-color: #2980B9;
    }
    /* Improve dataframe appearance */
    .dataframe {
        border-collapse: collapse;
        width: 100%;
    }
    .dataframe th {
        background-color: transparent;
        color: #2C3E50;
        font-weight: bold;
        text-align: left;
        padding: 8px;
        border: 1px solid #ddd;
    }
    .dataframe td {
        padding: 8px;
        border: 1px solid #ddd;
    }
    .dataframe tr:nth-child(even) {
        background-color: rgba(242, 242, 242, 0.1);
    }
    /* Master file upload section */
    .master-files-section {
        margin-top: 20px;
        padding: 15px;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
    }
    .master-file-header {
        font-weight: bold;
        color: #2C3E50;
        font-size: 1.1rem;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Text preprocessing functions
def preprocess_text(text, is_phone=False):
    """
    Enhanced preprocessing to handle variations in business names and phone numbers
    """
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    if is_phone:
        # For phone numbers, remove all non-digit characters
        text = re.sub(r'\D', '', text)
        # Remove leading zeros if present (to standardize format)
        text = text.lstrip('0')
        return text
    
    # For non-phone fields
    # Remove common business suffixes and abbreviations
    pattern = r'\b(' + '|'.join(BUSINESS_PATTERNS["suffixes"]) + r')\b'
    text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Standardize common abbreviations in addresses
    for abbr, full in ADDRESS_PATTERNS["unit_types"]:
        text = re.sub(r'\b' + abbr + r'\b', full, text, flags=re.IGNORECASE)
    
    # Standardize location names
    for abbr, full in ADDRESS_PATTERNS["location_types"]:
        text = re.sub(r'\b' + abbr + r'\b', full, text, flags=re.IGNORECASE)
    
    # Remove special characters but keep spaces
    text = re.sub(r"[^\w\s]", " ", text)
    
    # Handle empty strings after preprocessing
    if not text.strip():
        return ""
    
    # Normalize spaces and join words
    return " ".join(text.split())

def calculate_similarity_scores(str1, str2, is_phone=False):
    """
    Calculate similarity scores for a pair of strings.
    For phone numbers, returns None if not an exact match (to exclude from averaging).
    For other fields, returns fuzzy match scores.
    """
    if is_phone:
        # Skip comparison if either value is "971"
        if str1 == "971" or str2 == "971":
            return None
        # For phone numbers, only consider exact matches
        return 100 if str1 == str2 and str1 != "" else None
    
    # For non-phone fields, use enhanced fuzzy matching
    # First try exact match after preprocessing
    if str1 == str2:
        return 100
    
    # Calculate various similarity scores
    ratio = fuzz.ratio(str1, str2)
    partial_ratio = fuzz.partial_ratio(str1, str2)
    token_sort_ratio = fuzz.token_sort_ratio(str1, str2)
    token_set_ratio = fuzz.token_set_ratio(str1, str2)
    
    # Return the highest score among all methods
    return max(ratio, partial_ratio, token_sort_ratio, token_set_ratio)

def calculate_overlap_scores(str1, str2, is_phone=False):
    """
    Calculate overlap scores for a pair of strings.
    For phone numbers, returns None if not an exact match (to exclude from averaging).
    For other fields, returns overlap score.
    """
    if is_phone:
        # Skip comparison if either value is "971"
        if str1 == "971" or str2 == "971":
            return None
        # For phone numbers, only consider exact matches
        return 1.0 if str1 == str2 and str1 != "" else None
    
    return calculate_overlap(str1, str2)

def calculate_overlap(str1, str2):
    """
    Enhanced overlap calculation with better handling of variations
    """
    # Split into words and create sets
    words1 = set(str1.lower().split())
    words2 = set(str2.lower().split())
    
    # Calculate intersection and union
    intersection = words1 & words2
    union = words1 | words2
    
    # Special handling for single-word matches
    if len(words1) == 1 or len(words2) == 1:
        # If one is a single word, check if it's a subset of the other
        if words1.issubset(words2) or words2.issubset(words1):
            return 1.0
    
    # Calculate Jaccard similarity for overlap
    overlap = len(intersection) / len(union) if union else 0.0
    
    # Boost score if one set is fully contained in the other
    if words1.issubset(words2) or words2.issubset(words1):
        overlap = max(overlap, 0.8)  # Minimum 0.8 score for subset matches
    
    return overlap

# Core duplicate detection logic
def remove_fuzzy_duplicates(df, columns, similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD, overlap_threshold=DEFAULT_OVERLAP_THRESHOLD):
    """
    Remove fuzzy duplicates from a DataFrame based on selected columns.
    Name columns must meet the threshold for records to be considered duplicates.
    """
    # Reset index to ensure unique indices
    df = df.reset_index(drop=True)
    
    # Create preprocessed columns for each selected column
    for col in columns:
        is_phone = any(phone_term in col.lower() for phone_term in ['phone', 'mobile', 'cell', 'tel'])
        df[f'preprocessed_{col}'] = df[col].apply(lambda x: preprocess_text(x, is_phone=is_phone))
    
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
            
            # Calculate scores for each column
            name_scores = {}
            other_scores = {}
            name_overlaps = {}
            other_overlaps = {}
            
            for col in columns:
                str1 = df.at[i, f'preprocessed_{col}']
                str2 = df.at[j, f'preprocessed_{col}']
                
                # Check if this is a phone number column
                is_phone = any(phone_term in col.lower() for phone_term in ['phone', 'mobile', 'cell', 'tel'])
                # Check if this is a name column
                is_name = any(name_term in col.lower() for name_term in ['name', 'company', 'business', 'organization'])
                
                # Calculate similarity and overlap scores
                score = calculate_similarity_scores(str1, str2, is_phone)
                overlap = calculate_overlap_scores(str1, str2, is_phone)
                
                # Only include non-None scores (excludes non-matching phone numbers)
                if score is not None:
                    if is_name:
                        name_scores[col] = score
                    else:
                        other_scores[col] = score
                if overlap is not None:
                    if is_name:
                        name_overlaps[col] = overlap
                    else:
                        other_overlaps[col] = overlap
            
            # Calculate average scores for name and other columns
            avg_name_score = sum(name_scores.values()) / len(name_scores) if name_scores else 0
            avg_other_score = sum(other_scores.values()) / len(other_scores) if other_scores else 0
            avg_name_overlap = sum(name_overlaps.values()) / len(name_overlaps) if name_overlaps else 0
            avg_other_overlap = sum(other_overlaps.values()) / len(other_overlaps) if other_overlaps else 0
            
            # Consider as duplicate only if name columns meet the threshold
            # and we have valid scores for both name and other columns
            if (len(name_scores) > 0 and avg_name_score >= similarity_threshold and 
                avg_name_overlap >= overlap_threshold):
                # Only then check other columns
                if len(other_scores) == 0 or (avg_other_score >= similarity_threshold and 
                    avg_other_overlap >= overlap_threshold):
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
            
            # Calculate scores for reporting
            name_scores = {}
            other_scores = {}
            for col in columns:
                str1 = df.at[original_index, f'preprocessed_{col}']
                str2 = df.at[duplicate_index, f'preprocessed_{col}']
                is_phone = any(phone_term in col.lower() for phone_term in ['phone', 'mobile', 'cell', 'tel'])
                is_name = any(name_term in col.lower() for name_term in ['name', 'company', 'business', 'organization'])
                
                score = calculate_similarity_scores(str1, str2, is_phone)
                if score is not None:
                    if is_name:
                        name_scores[col] = score
                    else:
                        other_scores[col] = score
            
            # Combine scores for reporting
            scores = {**name_scores, **other_scores}
            
            # Calculate overall overlap
            overall_overlap = 0
            overlap_count = 0
            for col in columns:
                str1 = df.at[original_index, f'preprocessed_{col}']
                str2 = df.at[duplicate_index, f'preprocessed_{col}']
                is_phone = any(phone_term in col.lower() for phone_term in ['phone', 'mobile', 'cell', 'tel'])
                
                overlap = calculate_overlap_scores(str1, str2, is_phone)
                if overlap is not None:
                    overall_overlap += overlap
                    overlap_count += 1
            
            overall_overlap = overall_overlap / overlap_count if overlap_count > 0 else 0
            
            similarity_details[(duplicate_index, original_index)] = {
                "Name Scores": name_scores,
                "Other Scores": other_scores,
                "Overall Scores": scores,
                "Overlap": overall_overlap,
                "Actual Values": {col: {
                    "Duplicate Value": df.at[duplicate_index, col],
                    "Original Value": df.at[original_index, col]
                } for col in columns}
            }
    
    # Create a default unique dataframe that uses the first record of each duplicate group
    default_selected = {group[0] for group in groups}
    duplicates_all = set()
    for group in groups:
        duplicates_all.update(set(group))
    unique_indices = list(set(df.index) - (duplicates_all - default_selected))
    
    # Drop all preprocessed columns
    preprocessed_cols = [f'preprocessed_{col}' for col in columns]
    unique_df = df.loc[unique_indices].drop(columns=preprocessed_cols)
    
    duplicates_df = df.loc[list(duplicates_all - default_selected)].drop(columns=preprocessed_cols)
    # Add details for reporting
    duplicates_df['Duplicate Index'] = [d[0] for d in duplicates]
    duplicates_df['Original Index'] = [d[1] for d in duplicates]
    duplicates_df['Similarity Details'] = [similarity_details[d] for d in duplicates]
    
    return unique_df, duplicates_df, similarity_details, groups

# Function to check for duplicates against master files
def check_against_master_files(df, customer_df, columns, similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD, overlap_threshold=DEFAULT_OVERLAP_THRESHOLD):
    """
    Check for duplicates against master files with improved performance and user feedback.
    For customer master file, only exact columns "Address", "Customer Name", and "Main Phone Number" are used.
    """
    # Create a copy of the input DataFrame to add duplicate status
    result_df = df.copy()
    result_df['Duplicate Status'] = 'Not Duplicate'
    result_df['Master File Match'] = 'None'
    
    # Create containers for progress tracking
    status_container = st.empty()
    progress_container = st.empty()
    details_container = st.empty()
    eta_container = st.empty()
    
    # Function to create preprocessed columns efficiently
    def preprocess_dataframe_columns(df, columns_to_process):
        processed_df = df.copy()
        for col in columns_to_process:
            if col in df.columns:
                is_phone = any(phone_term in col.lower() for phone_term in ['phone', 'mobile', 'cell', 'tel'])
                processed_df[f'preprocessed_{col}'] = df[col].apply(lambda x: preprocess_text(x, is_phone=is_phone))
        return processed_df
    
    # Process input DataFrame
    status_container.info("📊 Preprocessing input data...")
    df_processed = preprocess_dataframe_columns(df, columns)
    
    # Process master files in batches
    BATCH_SIZE = 1000  # Adjust based on available memory
    
    def process_master_file(input_df, master_df, master_columns, file_name):
        if master_df is None or len(master_df) == 0:
            return
            
        status_container.info(f"🔍 Checking against {file_name}...")
        
        # Preprocess master file
        details_container.info(f"Preprocessing {file_name} columns...")
        master_processed = preprocess_dataframe_columns(master_df, master_columns)
        
        total_batches = (len(input_df) + BATCH_SIZE - 1) // BATCH_SIZE
        start_time = time.time()
        matches_found = 0
        
        for batch_idx in range(total_batches):
            # Calculate batch indices
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min((batch_idx + 1) * BATCH_SIZE, len(input_df))
            batch_df = input_df.iloc[start_idx:end_idx]
            
            # Update progress
            progress = (batch_idx + 1) / total_batches
            progress_container.progress(progress)
            
            # Calculate ETA
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / progress if progress > 0 else 0
            remaining_time = estimated_total_time - elapsed_time
            eta_container.info(f"⏱️ Estimated time remaining: {remaining_time:.1f} seconds")
            
            # Process batch
            details_container.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_df)} records)")
            
            # Vectorized comparison for the batch
            for idx, row in batch_df.iterrows():
                # Early exit if already marked as duplicate
                if result_df.at[idx, 'Duplicate Status'] == 'Duplicate':
                    continue
                
                # Calculate similarity scores for all columns in parallel
                match_found = False
                for master_idx, master_row in master_processed.iterrows():
                    name_scores = {}
                    other_scores = {}
                    name_overlaps = {}
                    other_overlaps = {}
                    
                    # For customer master file, only use the exact specified columns
                    if file_name == 'Customer Master File':
                        # Only use exact column matches for customer master file
                        customer_master_columns = ['Address', 'Customer Name', 'Main Phone Number']
                        
                        # Create mapping from input columns to customer master columns
                        column_mapping = {}
                        for col in columns:
                            # Only map columns that exactly match the required customer master columns
                            if col in customer_master_columns:
                                column_mapping[col] = col
                    else:
                        # For other master files, use direct column mapping
                        column_mapping = {col: col for col in columns if col in master_columns}
                    
                    for col in columns:
                        master_col = column_mapping.get(col)
                        
                        if master_col and master_col in master_processed.columns:
                            str1 = row[f'preprocessed_{col}']
                            str2 = master_row[f'preprocessed_{master_col}']
                            
                            is_phone = any(phone_term in col.lower() for phone_term in ['phone', 'mobile', 'cell', 'tel'])
                            is_name = any(name_term in col.lower() for name_term in ['name', 'company', 'business', 'organization'])
                            
                            if is_phone:
                                # Skip if either is "971"
                                if str1 == "971" or str2 == "971":
                                    continue
                                # For phone numbers, only consider exact matches
                                score = 100 if str1 == str2 and str1 != "" else 0
                                overlap = 1.0 if str1 == str2 and str1 != "" else 0.0
                            else:
                                # For non-phone fields, use enhanced fuzzy matching
                                score = fuzz.ratio(str1, str2)
                                overlap = calculate_overlap(str1, str2)
                            
                            if score >= similarity_threshold:
                                if is_name:
                                    name_scores[col] = score
                                    name_overlaps[col] = overlap
                                else:
                                    other_scores[col] = score
                                    other_overlaps[col] = overlap
                    
                    # Calculate average scores for name and other columns
                    avg_name_score = sum(name_scores.values()) / len(name_scores) if name_scores else 0
                    avg_name_overlap = sum(name_overlaps.values()) / len(name_overlaps) if name_overlaps else 0
                    
                    # Only consider as duplicate if name columns meet the threshold
                    if len(name_scores) > 0 and avg_name_score >= similarity_threshold and avg_name_overlap >= overlap_threshold:
                        # Then check other columns if they exist
                        if len(other_scores) == 0 or (
                            sum(other_scores.values()) / len(other_scores) >= similarity_threshold and
                            sum(other_overlaps.values()) / len(other_overlaps) >= overlap_threshold
                        ):
                            result_df.at[idx, 'Duplicate Status'] = 'Duplicate'
                            result_df.at[idx, 'Master File Match'] = file_name
                            matches_found += 1
                            match_found = True
                            break
                
                if matches_found > 0 and matches_found % 10 == 0:
                    details_container.info(f"Found {matches_found} matches in {file_name}")
    
    # Process customer master file - explicitly use only the three required columns with exact names
    customer_columns = ['Address', 'Customer Name', 'Main Phone Number']
    process_master_file(df_processed, customer_df, customer_columns, 'Customer Master File')
    
    # Clear progress indicators
    status_container.empty()
    progress_container.empty()
    details_container.empty()
    eta_container.empty()
    
    # Drop preprocessed columns
    preprocessed_cols = [col for col in result_df.columns if col.startswith('preprocessed_')]
    result_df = result_df.drop(columns=preprocessed_cols)
    
    return result_df

# Function to check for exact matches between customer duplicates and account master file
def check_account_master_matches(tagged_df, account_df):
    """
    Check for exact matches between customer duplicates and account master file.
    Only checks records that are duplicates against Customer Master File.
    Uses exact matching between Customer Name and Account Name fields.
    """
    if account_df is None or len(account_df) == 0:
        return tagged_df
    
    # Create a copy of the tagged dataframe
    result_df = tagged_df.copy()
    
    # Add Account Number and Last Shipment Date columns
    result_df['Account Number'] = None
    result_df['Last Shipment Date'] = None
    
    # Create containers for progress tracking
    status_container = st.empty()
    progress_container = st.empty()
    eta_container = st.empty()
    
    # Get only the records that are duplicates against Customer Master File
    customer_duplicates = result_df[result_df['Customer_DB_Status'] == 'Duplicate against Customer DB']
    
    if len(customer_duplicates) == 0:
        status_container.info("No customer duplicates found to check against Account Master File.")
        status_container.empty()
        return result_df
    
    status_container.info(f"🔍 Checking {len(customer_duplicates)} customer duplicates against Account Master File...")
    
    # Process in batches for better performance
    BATCH_SIZE = 1000
    total_batches = (len(customer_duplicates) + BATCH_SIZE - 1) // BATCH_SIZE
    start_time = time.time()
    matches_found = 0
    
    for batch_idx in range(total_batches):
        # Calculate batch indices
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, len(customer_duplicates))
        batch_indices = customer_duplicates.index[start_idx:end_idx]
        
        # Update progress
        progress = (batch_idx + 1) / total_batches
        progress_container.progress(progress)
        
        # Calculate ETA
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / progress if progress > 0 else 0
        remaining_time = estimated_total_time - elapsed_time
        eta_container.info(f"⏱️ Estimated time remaining: {remaining_time:.1f} seconds")
        
        # Process each duplicate record in the batch
        for idx in batch_indices:
            # Get the customer name for this record
            customer_name = tagged_df.at[idx, 'Customer Name'] if 'Customer Name' in tagged_df.columns else ""
            
            if customer_name and not pd.isna(customer_name):
                # Check for exact match with Account Name in account_df
                exact_matches = account_df[account_df['Account Name'].str.lower() == customer_name.lower()]
                
                if len(exact_matches) > 0:
                    # Get the first match
                    match = exact_matches.iloc[0]
                    
                    # Add Account Number and Last Shipment Date to the result
                    if 'Account Number' in match:
                        result_df.at[idx, 'Account Number'] = match['Account Number']
                    
                    if 'Last Shipment Date' in match:
                        result_df.at[idx, 'Last Shipment Date'] = match['Last Shipment Date']
                    
                    matches_found += 1
    
    # Clear progress indicators
    status_container.info(f"✅ Found {matches_found} matches in Account Master File.")
    progress_container.empty()
    eta_container.empty()
    
    return result_df

def generate_tagged_data(df, dup_groups, columns, similarity_details, customer_df=None, account_df=None):
    """
    Generate tagged data with duplicate status:
    - Original: The earliest record by Created Date or first record if dates are identical
    - Duplicate: Records identified as duplicates of an original
    - Not Duplicate: Records not in any duplicate group
    
    Also adds:
    - Similarity_Details column with JSON-formatted similarity and overlap scores
    - Customer_DB_Status column indicating if the record is a duplicate against Customer DB
    - Customer_DB_Details column with JSON-formatted similarity and overlap scores for customer DB matches
    - Account Number and Last Shipment Date columns for records matched with Account Master File
    """
    tagged_df = df.copy()
    tagged_df['Duplicate Status'] = 'Not Duplicate'
    tagged_df['Similarity_Details'] = None
    tagged_df['Customer_DB_Status'] = 'Not duplicate against Customer DB'
    tagged_df['Customer_DB_Details'] = None
    tagged_df['Account Number'] = None
    tagged_df['Last Shipment Date'] = None
    
    # Process each duplicate group
    for group in dup_groups:
        if len(group) > 1:  # If it's a duplicate group (has more than 1 record)
            # Get the subset of data for this group
            group_df = df.loc[group].copy()
            
            # Check if 'Created Date' column exists
            if 'Created Date' in group_df.columns:
                # Try to convert to datetime, with error handling
                try:
                    group_df['Created Date'] = pd.to_datetime(group_df['Created Date'])
                    # Sort by Created Date and get the earliest record's index
                    original_idx = group_df.sort_values('Created Date').index[0]
                except (ValueError, TypeError):
                    # If conversion fails, use the first record as original
                    original_idx = group[0]
            else:
                # If no Created Date column, use the first record as original
                original_idx = group[0]
            
            # Mark the original record
            tagged_df.at[original_idx, 'Duplicate Status'] = 'Original'
            
            # Mark all other records in the group as duplicates and add similarity details
            for idx in group:
                if idx != original_idx:
                    tagged_df.at[idx, 'Duplicate Status'] = 'Duplicate'
                    
                    # Get similarity details for this record
                    details = similarity_details.get((idx, original_idx), {})
                    if not details and similarity_details.get((original_idx, idx)):
                        # If details are stored with reversed indices, use those
                        details = similarity_details.get((original_idx, idx), {})
                    
                    # Create a simplified JSON structure with scores for each column
                    json_details = {}
                    
                    # Add similarity scores for each column
                    for col in columns:
                        col_score = None
                        col_overlap = None
                        
                        # Check in Name Scores for similarity
                        if col in details.get("Name Scores", {}):
                            col_score = details["Name Scores"][col]
                        # Check in Other Scores for similarity
                        elif col in details.get("Other Scores", {}):
                            col_score = details["Other Scores"][col]
                        
                        # Calculate overlap for this column
                        str1 = df.at[original_idx, col]
                        str2 = df.at[idx, col]
                        is_phone = any(phone_term in col.lower() for phone_term in ['phone', 'mobile', 'cell', 'tel'])
                        
                        # Preprocess the strings
                        str1_processed = preprocess_text(str1, is_phone=is_phone)
                        str2_processed = preprocess_text(str2, is_phone=is_phone)
                        
                        # Calculate overlap
                        col_overlap = calculate_overlap_scores(str1_processed, str2_processed, is_phone)
                        if col_overlap is None:
                            col_overlap = 0
                        
                        # Add to JSON if we have a score
                        if col_score is not None:
                            json_details[col] = {
                                "similarity": col_score,
                                "overlap": col_overlap,
                                "original_value": str(df.at[original_idx, col]),
                                "duplicate_value": str(df.at[idx, col])
                            }
                    
                    # Add overall overlap score
                    json_details["overall_overlap"] = details.get("Overlap", 0)
                    
                    # Store as JSON string
                    tagged_df.at[idx, 'Similarity_Details'] = json.dumps(json_details)
    
    # Check against customer master file if available
    if customer_df is not None and len(customer_df) > 0:
        # Use specified columns for customer master file check
        customer_columns = ['Address', 'Customer Name', 'Main Phone Number']
        
        # Only check columns that exist in both dataframes
        valid_columns = [col for col in customer_columns if col in df.columns and col in customer_df.columns]
        
        if valid_columns:
            st.info("Checking duplicates against Customer Master File...")
            # Implementing a simplified check here focusing on exact matches for Customer Name
            for idx, row in tagged_df.iterrows():
                if 'Customer Name' in valid_columns and 'Customer Name' in customer_df.columns:
                    customer_name = row.get('Customer Name', '')
                    if customer_name and not pd.isna(customer_name):
                        # Look for exact matches in customer master file
                        matches = customer_df[customer_df['Customer Name'].str.lower() == customer_name.lower()]
                        if len(matches) > 0:
                            tagged_df.at[idx, 'Customer_DB_Status'] = 'Duplicate against Customer DB'
                            
                            # Create details JSON
                            details = {
                                'Customer Name': {
                                    'similarity': 100,  # Exact match
                                    'original_value': customer_name,
                                    'master_value': matches.iloc[0]['Customer Name']
                                }
                            }
                            tagged_df.at[idx, 'Customer_DB_Details'] = json.dumps(details)
    
    # Check against account master file if available
    if account_df is not None and len(account_df) > 0:
        # Only proceed with account master file check for records that are duplicates against customer master file
        st.info("Checking customer duplicates against Account Master File...")
        customer_duplicates = tagged_df[tagged_df['Customer_DB_Status'] == 'Duplicate against Customer DB']
        
        if len(customer_duplicates) > 0 and 'Customer Name' in tagged_df.columns and 'Account Name' in account_df.columns:
            for idx in customer_duplicates.index:
                customer_name = tagged_df.at[idx, 'Customer Name']
                if customer_name and not pd.isna(customer_name):
                    # Look for exact matches in account master file
                    exact_matches = account_df[account_df['Account Name'].str.lower() == customer_name.lower()]
                    
                    if len(exact_matches) > 0:
                        # Get the first match
                        match = exact_matches.iloc[0]
                        
                        # Add Account Number and Last Shipment Date to the tagged dataframe
                        if 'Account Number' in match:
                            tagged_df.at[idx, 'Account Number'] = match['Account Number']
                        
                        if 'Last Shipment Date' in match:
                            tagged_df.at[idx, 'Last Shipment Date'] = match['Last Shipment Date']
    
    return tagged_df

def render_download_buttons(df, final_df=None, tagged_df=None, master_check_df=None):
    """Render download buttons for various dataframes"""
    if final_df is not None:
        # Only provide download button for final data without showing the table
        st.download_button(
            label="Download Final Unique Data as CSV",
            data=final_df.to_csv(index=False).encode("utf-8"),
            file_name="final_unique_data.csv",
            mime="text/csv",
        )
    
    if tagged_df is not None:
        st.write("Tagged Data (All Records with Duplicate Status):")
        
        # Display the dataframe directly
        st.dataframe(tagged_df)
        
        # Add a note about the JSON columns
        st.info("""
        - The 'Similarity_Details' column contains JSON-formatted similarity scores and overlap details for each duplicate record.
        - The 'Customer_DB_Status' column indicates if the record is a duplicate against the Customer DB.
        - The 'Customer_DB_Details' column contains JSON-formatted similarity scores and overlap details for matches against the Customer DB.
        - The 'Account Number' and 'Last Shipment Date' columns show data from the Account Master File for matching records.
        """)
        
        st.download_button(
            label="Download Tagged Data as CSV",
            data=tagged_df.to_csv(index=False).encode("utf-8"),
            file_name="tagged_data.csv",
            mime="text/csv",
            key="download_tagged"
        )
    
    if master_check_df is not None:
        # Only provide download button for master check results without showing the table
        st.download_button(
            label="Download Customer DB Match Results as CSV",
            data=master_check_df.to_csv(index=False).encode("utf-8"),
            file_name="customer_db_matches.csv",
            mime="text/csv",
            key="download_master_check"
        )

# Main application
def main():
    st.title("Fuzzy Duplicate Remover")
    st.write("Upload your Excel or CSV file to remove fuzzy duplicates.")
    
    # Load CSS
    load_css()

    # Use consistent keys in session state to preserve values on re-run
    if "reset" not in st.session_state:
        st.session_state.reset = False

    # File uploader with a dedicated key so the uploaded file remains available
    uploaded_file = st.file_uploader("Upload Excel or CSV file for duplicate check", type=["xlsx", "csv"], key="uploaded_file")
    
    # Master Files Section - with separate sections for each master file
    st.markdown('<div class="master-files-section">', unsafe_allow_html=True)
    
    # Customer Master File uploader
    st.markdown('<div class="master-file-header">Customer Master File (Optional)</div>', unsafe_allow_html=True)
    st.write("Upload a customer master file to check for duplicates against it.")
    customer_file = st.file_uploader("Upload Customer Master File", type=["xlsx", "csv"], key="customer_file")
    
    # Account Master File uploader
    st.markdown('<div class="master-file-header">Account Master File (Optional)</div>', unsafe_allow_html=True)
    st.write("Upload an account master file to match customer duplicates with account information.")
    account_file = st.file_uploader("Upload Account Master File", type=["xlsx", "csv"], key="account_file")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Load customer master file if uploaded
    if customer_file is not None and "customer_df" not in st.session_state:
        st.session_state.customer_df = load_data(customer_file)
        st.write("Customer Master File loaded successfully.")
    
    # Load account master file if uploaded
    if account_file is not None and "account_df" not in st.session_state:
        st.session_state.account_df = load_data(account_file)
        st.write("Account Master File loaded successfully.")
    
    if uploaded_file is not None:
        # Process the file on first upload and store the result in session_state
        if "df" not in st.session_state:
            st.session_state.df = load_data(uploaded_file)
        df = st.session_state.df

        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        # Allow user to select columns for duplicate check. Use a key to preserve selection.
        columns = st.multiselect("Select columns for duplicate check", df.columns, key="columns")
        if columns:
            similarity_threshold = st.slider("Similarity Score Threshold", 0, 100, DEFAULT_SIMILARITY_THRESHOLD, key="sim_thresh")
            overlap_threshold = st.slider("Word Overlap Threshold", 0.0, 1.0, DEFAULT_OVERLAP_THRESHOLD, key="overlap_thresh")

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
                st.write("Duplicates Detected:")
                st.dataframe(st.session_state.duplicates_df)

                # Automatically generate tagged data with duplicate status
                try:
                    customer_df = None
                    if "customer_df" in st.session_state:
                        customer_df = st.session_state.customer_df
                    else:
                        # If no customer master file is uploaded, use the input file itself
                        customer_df = df.copy()
                    
                    # Get account master file if available
                    account_df = None
                    if "account_df" in st.session_state:
                        account_df = st.session_state.account_df
                    
                    st.write("Processing tagged data...")
                    tagged_df = generate_tagged_data(
                        df, 
                        st.session_state.dup_groups, 
                        columns, 
                        st.session_state.similarity_details, 
                        customer_df,
                        account_df
                    )
                    
                    st.session_state.tagged_df = tagged_df
                    
                    # Create final dataframe with only Original and Not Duplicate records
                    final_df = tagged_df[tagged_df['Duplicate Status'] != 'Duplicate'].copy()
                    st.session_state.final_df = final_df
                    
                    # Check against master files if they are uploaded - this is now handled in generate_tagged_data
                    # We'll keep this section for the master_check_df which is used for download buttons
                    if "customer_df" in st.session_state:
                        master_check_df = tagged_df[['Customer_DB_Status', 'Customer_DB_Details']].copy()
                        if "account_df" in st.session_state and 'Account Number' in tagged_df.columns:
                            master_check_df['Account Number'] = tagged_df['Account Number']
                            master_check_df['Last Shipment Date'] = tagged_df['Last Shipment Date']
                        st.session_state.master_check_df = master_check_df
                    
                    # Render download buttons
                    st.write("Rendering tagged data...")
                    master_check_df = st.session_state.get("master_check_df")
                    render_download_buttons(df, final_df, tagged_df, master_check_df)
                except Exception as e:
                    st.error(f"Error processing tagged data: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

                # Add download button for duplicates
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
        st.rerun()

# UI Components
def load_data(uploaded_file):
    """Load data from uploaded file"""
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload an Excel or CSV file.")
        return None
    
    # Convert all columns to string to ensure consistent processing
    for col in df.columns:
        df[col] = df[col].astype(str)
    
    return df

if __name__ == "__main__":
    main()