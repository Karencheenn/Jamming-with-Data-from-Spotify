import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import random

# Loading dataset
# Dataset source: Top Spotify Songs from 2010-2019 - Compiled by Leonardo Henrique
# Available at: https://www.kaggle.com/datasets/leonardopena/top-spotify-songs-from-20102019-by-year

data = pd.read_csv("Spotify_music_data.csv")

# Find the 30 most popular entries based on the 'pop' column
top_songs = data.nlargest(30, 'pop')

# Generate text for the word cloud: repeat each title by its popularity
text = " ".join(f"{row['title']} " * row['pop'] for index, row in top_songs.iterrows())

# Dimensions of the canvas
width, height = 800, 400

# Create an ellipse mask
x, y = np.ogrid[:height, :width]
mask = (x - height / 2) ** 2 / (0.35 * height) ** 2 + (y - width / 2) ** 2 / (0.48 * width) ** 2 <= 1
mask = 255 * (~mask.astype(bool))  # Invert mask: inside the ellipse should be True/white

# Define a function to choose random colors
def random_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    h = int(random.random() * 360)  # Hue between 0 and 360
    s = int(100)  # Saturation at 100% for vivid colors
    l = int(random.random() * 30 + 40)  # Lightness between 40% and 70%
    return "hsl({}, {}%, {}%)".format(h, s, l)


# Create the word cloud with random colors and the elliptical mask
wordcloud = WordCloud(width=width, height=height, background_color='white', mask=mask, color_func=random_color_func, max_font_size=40, prefer_horizontal=1.0).generate(text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("Top 30 Popular Songs on Spotify from 2010 to 2019, measured by Billboard",fontsize=15, color='black')
plt.axis("off")
plt.show()


# Define BPM ranges
bpm_bins = [0, 80, 120, 160, 200]  # Modify these ranges as suitable
bpm_labels = ['0-80', '81-120', '121-160', '161-200']
data['bpm_category'] = pd.cut(data['bpm'], bins=bpm_bins, labels=bpm_labels, right=False)

# Group by 'top genre' and 'bpm_category' and count occurrences
genre_bpm_counts = data.groupby(['top genre', 'bpm_category']).size().unstack(fill_value=0)

# Plotting
ax = genre_bpm_counts.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
plt.title('Number of Songs per Top Genre by BPM')
plt.xlabel('Top Genre')
plt.ylabel('Number of Songs')
plt.xticks(rotation=45)  # Rotate genre labels for better readability
plt.legend(title='BPM Categories')  # Ensure the legend is visible with a title
plt.show()