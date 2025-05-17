import os
import re
from collections import Counter

# Directory containing the text files
dir_path = os.path.join(os.path.dirname(__file__), 'activity_data_text')

# Collect all .txt files
files = [f for f in os.listdir(dir_path) if f.endswith('.txt')]

unigram_counter = Counter()

for filename in files:
    with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as file:
        text = file.read().lower()
        # Tokenize: split on non-word characters
        words = re.findall(r'\b\w+\b', text)
        unigram_counter.update(words)

# Write to unigram.dic
with open(os.path.join(os.path.dirname(__file__), 'unigram.dic'), 'w', encoding='utf-8') as out:
    for word, freq in unigram_counter.most_common():
        out.write(f'{word} {freq}\n')

print(f"Unigram dictionary built with {len(unigram_counter)} unique words.")
