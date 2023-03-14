#Importing Libraries
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
import cv2

def transform_format(val):
    if val == 180 or val == 183 or val == 181:
        return 255
    else:
        return val

stopwords = set(STOPWORDS)
#Importing Dataset
df = pd.read_csv("MALratings.csv")
df2 = df[df.Rank != "/A"]
df2.Rank = pd.to_numeric(df2.Rank, errors = 'coerce')
df3 = df2.sort_values(by = "Rank", ascending = True)

mask = np.array(Image.open("Untitled.png"))
mask
'''for i in range(len(mask)):
   #mask[i] = list(map(transform_format, mask[i]))
    for j in range(len(mask[i])):
        mask[i][j] = list(map(transform_format, mask[i][j]))'''
mask
#Checking the Data

#Creating the text variable
text = " ".join(title for title in df.Title)

# Creating word_cloud with text as argument in .generate() method
word_cloud = WordCloud(max_font_size=30, max_words=10000,  collocations=False, background_color='black', mask=mask, \
             contour_width=1, contour_color="grey", stopwords=stopwords).generate(text)

# Display and save the generated Word Cloud
# wordcloud.to_file("img/first_review.png")
image_colors = ImageColorGenerator(mask)
plt.figure(figsize=[10, 10])
plt.imshow(word_cloud.recolor(color_func=image_colors), interpolation='bilinear')
plt.axis("off")
plt.savefig("saved.png", format="png")
plt.show()
