# install pandas and spacy library

import pandas as pd
#review the spacy NLP library
import spacy
import re
from top2vec import Top2Vec
from gensim import corpora
from gensim.models import LdaModel

with open("bain_fashion.txt", "r") as file:
    document = file.readlines()
with open("bain_coffee.txt", "r") as file:
    document += file.readlines()

# Preprocess the text data
def preprocess(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    return text.lower()

processed_docs = [preprocess(doc).split() for doc in document]

# Step 1: Load the text data from the file
with open("bain_fashion.txt", "r") as file:
    document = file.readlines()
with open("bain_coffee.txt", "r") as file:
    document += file.readlines()


# Step 2: Train the Top2Vec model
model = Top2Vec(document, speed="learn", workers=4, min_count=1)

# Step 3: Extract the topics
topics = []
num_topics = model.get_num_topics()

for topic_num in range(num_topics):
    words, word_scores = model.get_topic(topic_num)
    topic_info = {
        "Topic Number": topic_num,
        "Words": ", ".join(words),
    }
    topics.append(topic_info)

# Step 4: Save the topics to a CSV file
df = pd.DataFrame(topics)
df.to_csv("topics.csv", index=False)

print("Topics saved to topics.csv")
'''

'''
# Alternative approach using LDA with gensim
with open("bain_fashion.txt", "r") as file:
    document = file.readlines()
with open("bain_coffee.txt", "r") as file:
    document += file.readlines()

# Preprocess the text data
def preprocess(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    return text.lower()

processed_docs = [preprocess(doc).split() for doc in document]

# Create a dictionary and corpus for LDA
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Train the LDA model
lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

# Extract the topics
topics = []
for idx, topic in lda_model.print_topics(-1):
    topic_info = {
        "Topic Number": idx,
        "Words": topic,
    }
    topics.append(topic_info)

# Save the topics to a CSV file
df = pd.DataFrame(topics)
df.to_csv("topics.csv", index=False)

print("Topics saved to topics.csv")

# Search and sort topics from the CSV file
def search_and_sort_topics(csv_file, search_terms):
    df = pd.read_csv(csv_file)
    filtered_df = df[df['Words'].str.contains('|'.join(search_terms), case=False)]
    sorted_df = filtered_df.sort_values(by='Topic Number')
    return sorted_df

# Example usage
search_terms = ['company', 'location','idea']
sorted_topics = search_and_sort_topics("topics.csv", search_terms)
print(sorted_topics)