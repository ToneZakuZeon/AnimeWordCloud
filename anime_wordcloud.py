import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS


#removes unimportant words like (we, i, the, etc)
stopwords = set(STOPWORDS)

#importing dataset to use for the wordcloud
df = pd.read_csv('datasets/MALratings.csv')

#excludes anime that have non-applicable ranking
df2 = df[df.Rank != "/A"]
df2['Rank'] = df2['Rank'].astype(int)
anime = df2.sort_values(by ="Rank", ascending=True)
len(anime)
del df
del df2

anime = anime.head(10000) #change number based on how much anime you want included, using more data means more visible image for larger masks

text = " ".join(title for title in anime.Title) #using the Title column of the dataset for the wordcloud

#input image file path to use ask your mask for a wordcloud
img = Image.open("images/bocchi.png").convert("RGB")
mask = np.array(img)

#adjust the wordcloud here, if the font is too small then less popular words won't appear,
#if the font is too big than words might not fit the shape, higher max words allows for more detail at the cost of time
#changing background_color to another color can make a wordcloud be more or less visible
#change contour_width if you want an outline drawn around the image with a color of contour_color

word_cloud = WordCloud(max_font_size= 30, collocations=False,
background_color="black", max_words = 9000, stopwords=stopwords, contour_width=0, contour_color="grey",  mask = mask).generate(text)

image_colors = ImageColorGenerator(mask)
#adjust fig size to image resolution, ex: 1280x720 will be [12.8, 7.2]
plt.figure(figsize=[12.8, 7.2])

#uses the colors that are found in the image
plt.imshow(word_cloud.recolor(color_func=image_colors), interpolation='bilinear')

#uses the the default colors, might want to use if the colors are only in the mask image are grey-scale
#plt.imshow(word_cloud, interpolation='bilinear', aspect="auto")

plt.axis("off")
#removes empty white borders
plt.tight_layout(pad=0)
plt.savefig("BocchiCloud.png", format="png")
plt.show()

