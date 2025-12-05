import pandas as pd
import re
import os

# =============================================================================
# STEP 1: Extract emojis and descriptions from LLM CSV files and combine with Human response
# =============================================================================

# Read the Human response CSV file
human_df = pd.read_csv('Human response.csv')

# List of LLM CSV files and their corresponding column names
llm_files = [
    'Qwen2.5-1.5B.csv',
    'Qwen2.5-14B.csv', 
    'gemma-3-1b.csv',
    'Qwen2.5-7B.csv',
    'gemma-3-4b.csv',
    'Qwen2.5-3B.csv',
    'Yi-1.5-6B.csv',
    'Yi-1.5-9B.csv'
]

# Function to extract emojis and descriptions from response
def extract_emojis_and_description(text):
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Extract emojis using regex (Unicode emoji range)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # other symbols
        "\U000024C2-\U0001F251"  # enclosed characters
        "]+", 
        flags=re.UNICODE
    )
    
    emojis = ''.join(emoji_pattern.findall(text))
    
    # Extract description (text without emojis)
    description = emoji_pattern.sub('', text).strip()
    
    # Clean up description - remove extra spaces and common prefixes
    description = re.sub(r'\s+', ' ', description)
    description = re.sub(r'^[:\-]\s*', '', description)
    
    # Combine emojis and description
    if emojis and description:
        return f"{emojis} - {description}"
    elif emojis:
        return emojis
    elif description:
        return description
    else:
        return ""

# Process each LLM file and add as a new column
for llm_file in llm_files:
    if os.path.exists(llm_file):
        # Read the LLM CSV file
        llm_df = pd.read_csv(llm_file)
        
        # Extract the base name for column naming (remove .csv extension)
        col_name = llm_file.replace('.csv', '')
        
        # Apply the extraction function to the response column
        human_df[col_name] = llm_df['response'].apply(extract_emojis_and_description)
        
        print(f"Processed {llm_file} -> Added column '{col_name}'")
    else:
        print(f"Warning: {llm_file} not found")

# Save the intermediate combined CSV
human_df.to_csv('Human_response_with_LLMs_combined.csv', index=False)
print("\nStep 1 completed: Combined CSV saved as 'Human_response_with_LLMs_combined.csv'")

# =============================================================================
# STEP 2: Remove all text from first 10 responses, keeping only emojis
# =============================================================================

# Read the combined CSV file
df = pd.read_csv('Human_response_with_LLMs_combined.csv')

# List of LLM columns (excluding 'Question' and 'Human Response')
llm_columns = [col for col in df.columns if col not in ['Question', 'Human Response']]

# Function to extract only emojis from text (remove ALL other characters)
def extract_only_emojis(text):
    if pd.isna(text) or text == "":
        return ""
    
    text = str(text)
    
    # Extract emojis using regex (Unicode emoji range)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # other symbols
        "\U000024C2-\U0001F251"  # enclosed characters
        "]+", 
        flags=re.UNICODE
    )
    
    # Find all emojis
    emojis = emoji_pattern.findall(text)
    
    # Join all emojis together and remove ANY non-emoji characters
    emoji_only = ''.join(emojis)
    
    return emoji_only

# Apply the function to first 10 rows of each LLM column
for col in llm_columns:
    # Apply to first 10 rows only (emoji selection tasks)
    df.loc[:9, col] = df.loc[:9, col].apply(extract_only_emojis)

# Save the final CSV
df.to_csv('Human_response_final.csv', index=False)

print("\nStep 2 completed: Final CSV saved as 'Human_response_final.csv'")
print("First 10 rows of LLM columns now contain ONLY emojis (all text removed)")
print("Rows 11+ retain original emojis + descriptions for sentiment analysis")

# Display summary
print(f"\nFinal DataFrame shape: {df.shape}")
print("Columns:", list(df.columns))
print("\nSample of first 3 rows:")
print(df.head(3))