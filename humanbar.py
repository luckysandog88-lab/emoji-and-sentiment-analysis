import pandas as pd
import matplotlib.pyplot as plt
import emoji


# Read files
emoji_categories_df = pd.read_csv('emoji_categories.csv', encoding='utf-8-sig')
human_response_df = pd.read_csv('1-10only.csv', encoding='utf-8-sig')


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


print(f"Found {len(emotion_base_set)} unique emotion emojis")
print(f"Found {len(concrete_base_set)} unique concrete emojis")


# Find response column
response_col = next((col for col in human_response_df.columns
                    if 'response' in col.lower()), human_response_df.columns[0])


print(f"Analyzing column: '{response_col}'")


# Count matches
emotion_count = 0
concrete_count = 0
other_count = 0
all_emoji_sequences = []


for response in human_response_df[response_col].dropna():
   emoji_sequences = extract_complete_emojis(response)
   all_emoji_sequences.extend(emoji_sequences)
  
   for emoji_seq in emoji_sequences:
       base_emoji = get_base_emoji(emoji_seq)
      
       if base_emoji in emotion_base_set:
           emotion_count += 1
       elif base_emoji in concrete_base_set:
           concrete_count += 1
       else:
           other_count += 1


print(f"\n=== Results ===")
print(f"Total emoji sequences found: {len(all_emoji_sequences)}")
print(f"Emotion category matches: {emotion_count}")
print(f"Concrete category matches: {concrete_count}")
print(f"Other emojis: {other_count}")


# ====== CREATE CHART WITH SMALLER SIDE ANNOTATION BOX ======
# Set up the figure with a wider width to accommodate side annotation
fig, ax = plt.subplots(figsize=(11, 7))  # Slightly smaller figure


# Data for bars
categories = ['Emotion Category', 'Non-emotion Category']
counts = [emotion_count, concrete_count]
colors = ['#FF6B6B', '#4ECDC4']


# Create bars with enhanced styling
bars = ax.bar(categories, counts,
             color=colors,
             width=0.6,
             edgecolor=['#D95D5D', '#3EB7AF'],  # Slightly darker edges
             linewidth=2,
             zorder=3)


# Add value labels on top of bars
for bar, count in zip(bars, counts):
   height = bar.get_height()
   percentage = (count / len(all_emoji_sequences)) * 100 if all_emoji_sequences else 0
  
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
ax.set_title('Emoji Category Distribution in Human Responses',
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
# Create a separate axis for the side box - SMALLER size
side_ax = fig.add_axes([0.80, 0.75, 0.05, 0.05])  # SMALLER: [left, bottom, width, height]


# Turn off the axes for the side box
side_ax.axis('off')


# Create the side annotation box with total count - simpler text
total_text = f'TOTAL EMOJIS\nANALYZED:\n{len(all_emoji_sequences)}'


# Create a clean, smaller box
side_ax.text(0.5, 0.5, total_text,
            ha='center', va='center',
            fontsize=12, fontweight='bold',  # Smaller font
            color='#2C3E50',
            transform=side_ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.8",  # Less padding
                     facecolor="#F8F9FA",
                     edgecolor="#4ECDC4",
                     linewidth=2,  # Thinner border
                     alpha=0.95))


# Add a very subtle shadow effect
shadow_box = dict(boxstyle="round,pad=0.8",
                 facecolor="black",
                 edgecolor="black",
                 alpha=0.05)  # More subtle shadow
side_ax.text(0.51, 0.49, total_text,
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            color='#2C3E50',
            transform=side_ax.transAxes,
            bbox=shadow_box,
            zorder=0)


# Adjust main plot area to make room for side annotation
plt.subplots_adjust(right=0.8)  # Less space on the right


plt.tight_layout(rect=[0, 0, 0.85, 0.95])  # Adjust for smaller box


# Save with high quality
plt.savefig('emoji_analysis_small_total_box.png',
           dpi=300,
           bbox_inches='tight',
           facecolor='white',
           edgecolor='none')


print("\n✨ Chart with small total box saved as 'emoji_analysis_small_total_box.png'")
plt.show()

