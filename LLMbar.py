import pandas as pd
import matplotlib.pyplot as plt
import emoji


# Read files
emoji_categories_df = pd.read_csv('emoji_categories.csv', encoding='utf-8-sig')
human_response_df = pd.read_csv('1-10only.csv', encoding='utf-8-sig')


# List of model response columns to analyze
model_columns = [
   'Qwen2.5-1.5B',
   'Qwen2.5-14B',
   'gemma-3-1b',
   'Qwen2.5-7B',
   'gemma-3-4b',
   'Qwen2.5-3B',
   'Yi-1.5-6B',
   'Yi-1.5-9B'
]


# Check which columns actually exist in the CSV file
available_columns = []
for col in model_columns:
   if col in human_response_df.columns:
       available_columns.append(col)
   else:
       print(f"Warning: Column '{col}' not found in the CSV file.")


if not available_columns:
   print("No model response columns found. Available columns are:")
   print(human_response_df.columns.tolist())
   # Use all columns except the first one (assuming first is some ID or prompt column)
   available_columns = human_response_df.columns[1:].tolist()
   print(f"Using available columns: {available_columns}")


print(f"\nAnalyzing {len(available_columns)} model response columns: {available_columns}")


# Function to extract complete emoji sequences (including skin tones as part of emoji)
def extract_complete_emojis(text):
   """Extract emoji sequences, treating skin tones and ZWJ sequences as single emojis"""
   if pd.isna(text):
       return []
  
   text_str = str(text)
   emoji_sequences = []
   i = 0
  
   while i < len(text_str):
       char = text_str[i]
      
       # Check if current character is a base emoji
       if char in emoji.EMOJI_DATA:
           # Start building a sequence
           sequence = char
           i += 1
          
           # Look ahead for skin tones or ZWJ sequences
           while i < len(text_str):
               next_char = text_str[i]
              
               # Skin tone modifier (🏻🏼🏽🏾🏿)
               if next_char in ['🏻', '🏼', '🏽', '🏾', '🏿']:
                   sequence += next_char
                   i += 1
               # Zero-width joiner (for complex emojis like 🙇🏻‍♀️)
               elif next_char == '\u200d' and i + 1 < len(text_str):
                   # Check if next character after ZWJ is also an emoji
                   if text_str[i + 1] in emoji.EMOJI_DATA:
                       sequence += next_char + text_str[i + 1]
                       i += 2
                   else:
                       break
               # Variation selector (makes emoji colorful vs black/white)
               elif next_char == '\ufe0f':
                   sequence += next_char
                   i += 1
               else:
                   break
          
           emoji_sequences.append(sequence)
       else:
           i += 1
  
   return emoji_sequences


# Function to get base emoji (remove skin tones and modifiers for comparison)
def get_base_emoji(emoji_sequence):
   """Extract just the base emoji character from a sequence"""
   if not emoji_sequence:
       return ''
  
   # Remove skin tones
   skin_tones = ['🏻', '🏼', '🏽', '🏾', '🏿']
   base = emoji_sequence
   for tone in skin_tones:
       base = base.replace(tone, '')
  
   # Remove variation selector
   base = base.replace('\ufe0f', '')
  
   # For ZWJ sequences, take the first emoji component
   if '\u200d' in base:
       # Take everything before ZWJ or first character
       parts = base.split('\u200d')
       if parts[0]:
           return parts[0]
       elif len(parts) > 1 and parts[1]:
           return parts[1][0] if parts[1] else ''
  
   # Return first character (should be the base emoji)
   return base[0] if base else ''


# Get emoji sets from categories (using base emojis for comparison)
emotion_categories = extract_complete_emojis(emoji_categories_df.iloc[0, 0])
concrete_categories = extract_complete_emojis(emoji_categories_df.iloc[0, 1])


# Create sets of base emojis for comparison
emotion_base_set = {get_base_emoji(e) for e in emotion_categories if get_base_emoji(e)}
concrete_base_set = {get_base_emoji(e) for e in concrete_categories if get_base_emoji(e)}


# Combine both sets for easy lookup
all_categories_set = emotion_base_set.union(concrete_base_set)


print(f"Found {len(emotion_base_set)} unique emotion emojis")
print(f"Found {len(concrete_base_set)} unique concrete emojis")
print(f"Total unique emojis in categories: {len(all_categories_set)}")


# ====== ANALYZE ALL MODEL RESPONSE COLUMNS ======
print("\n" + "="*60)
print("ANALYZING MODEL RESPONSE COLUMNS")
print("="*60)


# Initialize counters for ALL models combined
total_emotion_count = 0
total_concrete_count = 0
total_other_count = 0
total_all_emoji_sequences = []


# Dictionary to store NOT IN CATEGORY emojis
not_in_category_emojis = {}
all_found_emojis = set()


# Dictionary to store results for each model
model_results = {}


for model_col in available_columns:
   print(f"\nAnalyzing column: '{model_col}'")
  
   # Initialize counters for this model
   model_emotion_count = 0
   model_concrete_count = 0
   model_other_count = 0
   model_all_emoji_sequences = []
   model_not_in_category = []
  
   for response in human_response_df[model_col].dropna():
       emoji_sequences = extract_complete_emojis(response)
       model_all_emoji_sequences.extend(emoji_sequences)
      
       for emoji_seq in emoji_sequences:
           base_emoji = get_base_emoji(emoji_seq)
          
           if base_emoji in emotion_base_set:
               model_emotion_count += 1
               total_emotion_count += 1
           elif base_emoji in concrete_base_set:
               model_concrete_count += 1
               total_concrete_count += 1
           else:
               model_other_count += 1
               total_other_count += 1
               model_not_in_category.append((emoji_seq, base_emoji))
          
           # Add to all found emojis set
           all_found_emojis.add(base_emoji)
  
   # Add to total sequences
   total_all_emoji_sequences.extend(model_all_emoji_sequences)
  
   # Store results for this model
   model_results[model_col] = {
       'total_emojis': len(model_all_emoji_sequences),
       'emotion_count': model_emotion_count,
       'concrete_count': model_concrete_count,
       'other_count': model_other_count,
       'not_in_category': model_not_in_category
   }
  
   # Store not-in-category emojis
   not_in_category_emojis[model_col] = model_not_in_category
  
   print(f"  Total emojis found: {len(model_all_emoji_sequences)}")
   print(f"  Emotion category matches: {model_emotion_count}")
   print(f"  Concrete category matches: {model_concrete_count}")
   print(f"  Other emojis (not in categories): {model_other_count}")


print(f"\n" + "="*60)
print("COMBINED RESULTS (All Models)")
print("="*60)
print(f"Total emoji sequences found: {len(total_all_emoji_sequences)}")
print(f"Total Emotion category matches: {total_emotion_count}")
print(f"Total Concrete category matches: {total_concrete_count}")
print(f"Total Other emojis: {total_other_count}")


# ====== IDENTIFY EMOJIS NOT IN CATEGORIES ======
print("\n" + "="*60)
print("EMOJIS NOT IN EMOJI_CATEGORIES.CSV")
print("="*60)


# Get all unique base emojis from responses
all_base_emojis = all_found_emojis


# Find which emojis are NOT in the categories
not_in_categories = all_base_emojis - all_categories_set


print(f"\nFound {len(not_in_categories)} unique emojis that are NOT in Emotion or Concrete categories:")


if not_in_categories:
   # Display the emojis
   for i, emoji_char in enumerate(sorted(not_in_categories, key=lambda x: ord(x))):
       print(f"{i+1:3d}. {emoji_char} (U+{ord(emoji_char):04X})")
  
   # Count frequency across all models
   print(f"\nFrequency of 'Not in Category' emojis across all models:")
   from collections import Counter
   all_not_in_category_base = []
  
   for model_col, not_in_list in not_in_category_emojis.items():
       for emoji_seq, base_emoji in not_in_list:
           if base_emoji in not_in_categories:
               all_not_in_category_base.append(base_emoji)
  
   if all_not_in_category_base:
       frequency_count = Counter(all_not_in_category_base)
       for emoji_char, count in sorted(frequency_count.items(), key=lambda x: x[1], reverse=True):
           print(f"  {emoji_char}: {count} times (U+{ord(emoji_char):04X})")
  
   # Show examples from each model
   print(f"\nExamples of responses containing 'Not in Category' emojis:")
   sample_shown = 0
   for model_col in available_columns:
       not_in_list = not_in_category_emojis[model_col]
       if not_in_list and sample_shown < 5:  # Show max 5 examples
           # Get first occurrence
           emoji_seq, base_emoji = not_in_list[0]
          
           # Find the actual response containing this emoji
           for idx, response in enumerate(human_response_df[model_col].dropna()):
               if emoji_seq in str(response):
                   response_preview = str(response)[:50] + "..." if len(str(response)) > 50 else str(response)
                   print(f"  {model_col}: '{emoji_seq}' in: '{response_preview}'")
                   sample_shown += 1
                   break
else:
   print("All emojis found in responses are in Emotion or Concrete categories!")


# ====== CREATE DETAILED CSV OF NOT-IN-CATEGORY EMOJIS ======
print(f"\n" + "="*60)
print("CREATING DETAILED ANALYSIS FILES")
print("="*60)


# Create detailed DataFrame for not-in-category emojis
detailed_records = []
for model_col in available_columns:
   not_in_list = not_in_category_emojis[model_col]
   for emoji_seq, base_emoji in not_in_list:
       # Find the actual response
       found_response = None
       for response in human_response_df[model_col].dropna():
           if emoji_seq in str(response):
               found_response = str(response)
               break
      
       detailed_records.append({
           'Model': model_col,
           'Emoji_Sequence': emoji_seq,
           'Base_Emoji': base_emoji,
           'Base_Unicode': f"U+{ord(base_emoji):04X}" if base_emoji else '',
           'In_Emotion_Category': base_emoji in emotion_base_set,
           'In_Concrete_Category': base_emoji in concrete_base_set,
           'Response_Text': found_response[:100] + "..." if found_response and len(found_response) > 100 else found_response if found_response else ''
       })


if detailed_records:
   detailed_df = pd.DataFrame(detailed_records)
   detailed_df.to_csv('emojis_not_in_categories_detailed.csv', index=False, encoding='utf-8-sig')
   print(f"Detailed analysis saved to 'emojis_not_in_categories_detailed.csv'")
  
   # Also save a summary by emoji
   summary_by_emoji = detailed_df.groupby(['Base_Emoji', 'Base_Unicode']).agg({
       'Model': lambda x: ', '.join(sorted(set(x))),
       'Response_Text': 'count'
   }).rename(columns={'Response_Text': 'Occurrence_Count'}).reset_index()
  
   summary_by_emoji = summary_by_emoji.sort_values('Occurrence_Count', ascending=False)
   summary_by_emoji.to_csv('emojis_not_in_categories_summary.csv', index=False, encoding='utf-8-sig')
   print(f"Summary by emoji saved to 'emojis_not_in_categories_summary.csv'")
  
   print(f"\nTop 10 most frequent 'Not in Category' emojis:")
   for i, row in summary_by_emoji.head(10).iterrows():
       print(f"  {row['Base_Emoji']}: {row['Occurrence_Count']} times in {row['Model']}")


# ====== CREATE CHART FOR COMBINED RESULTS ======
fig, ax = plt.subplots(figsize=(11, 7))


categories = ['Emotion Category', 'Non-emotion Category']
counts = [total_emotion_count, total_concrete_count]
colors = ['#FF6B6B', '#4ECDC4']


bars = ax.bar(categories, counts,
             color=colors,
             width=0.6,
             edgecolor=['#D95D5D', '#3EB7AF'],
             linewidth=2,
             zorder=3)


# Add value labels on top of bars
for bar, count in zip(bars, counts):
   height = bar.get_height()
   percentage = (count / len(total_all_emoji_sequences)) * 100 if total_all_emoji_sequences else 0
  
   # Add count (large and bold)
   ax.text(bar.get_x() + bar.get_width()/2, height + (max(counts)*0.02 if counts else 5),
           f'{count}',
           ha='center', va='bottom',
           fontsize=22, fontweight='bold', color='#2C3E50',
           zorder=4)
  
   # Add percentage below count
   ax.text(bar.get_x() + bar.get_width()/2, height * 0.7,
           f'({percentage:.1f}%)',
           ha='center', va='center',
           fontsize=14, fontweight='medium', color='#2C3E50',
           alpha=0.8, zorder=4)


# Customize the main chart
title_text = 'Emoji Category Distribution in LLM Responses\n'
if not_in_categories:
   title_text += f'({len(not_in_categories)} unique emojis not in categories)'


ax.set_title(title_text,
            fontsize=18, fontweight='bold', pad=25, color='#2C3E50')
ax.set_xlabel('Category', fontsize=14, fontweight='semibold', color='#2C3E50', labelpad=15)
ax.set_ylabel('Number of Emojis', fontsize=14, fontweight='semibold', color='#2C3E50', labelpad=15)


# Customize ticks
ax.tick_params(axis='x', labelsize=12, colors='#2C3E50')
ax.tick_params(axis='y', labelsize=11, colors='#2C3E50')


# Remove top and right spines for cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CCCCCC')
ax.spines['bottom'].set_color('#CCCCCC')


# Set y-axis limit with some padding
if counts:
   ax.set_ylim(0, max(counts) * 1.25)


# Add grid (only horizontal)
ax.yaxis.grid(True, color='#EEEEEE', linewidth=1, linestyle='-', alpha=0.7)
ax.xaxis.grid(False)


# ====== ADD SMALLER SIDE ANNOTATION BOX ======
side_ax = fig.add_axes([0.80, 0.75, 0.05, 0.05])
side_ax.axis('off')


total_text = f'TOTAL EMOJIS\nANALYZED:\n{len(total_all_emoji_sequences)}'


side_ax.text(0.5, 0.5, total_text,
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            color='#2C3E50',
            transform=side_ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.8",
                     facecolor="#F8F9FA",
                     edgecolor="#4ECDC4",
                     linewidth=2,
                     alpha=0.95))


shadow_box = dict(boxstyle="round,pad=0.8",
                 facecolor="black",
                 edgecolor="black",
                 alpha=0.05)
side_ax.text(0.51, 0.49, total_text,
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            color='#2C3E50',
            transform=side_ax.transAxes,
            bbox=shadow_box,
            zorder=0)


# Adjust layout
plt.subplots_adjust(right=0.8)
plt.tight_layout(rect=[0, 0, 0.85, 0.95])


plt.savefig('emoji_analysis_all_models_with_not_in_category.png',
           dpi=300,
           bbox_inches='tight',
           facecolor='white',
           edgecolor='none')


print(f"\n✨ Combined chart saved as 'emoji_analysis_all_models_with_not_in_category.png'")
plt.show()


print(f"\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"Files created:")
print(f"1. 'emojis_not_in_categories_detailed.csv' - Detailed list of emojis not in categories")
print(f"2. 'emojis_not_in_categories_summary.csv' - Summary by emoji")
print(f"3. 'emoji_analysis_all_models_with_not_in_category.png' - Visualization")

