

# before using this install the below modules
#pip install wikipedia==1.4.0

from wordcloud import WordCloud, STOPWORDS
import wikipedia
from PIL import Image

search=input("search something")
stop_w = set(STOPWORDS)
info = wikipedia.summary(search)
word_cloud = WordCloud(stopwords = stop_w).generate(info)
img = word_cloud.to_image()
img.show()
print(info)