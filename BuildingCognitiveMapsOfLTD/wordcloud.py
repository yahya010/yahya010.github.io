import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS
import pandas as pd

############ User Input #########################

#wcFolder = # input path to folder to save word cloud images. Include '/' at end. 'Data_output/'
wcFolder = '/Data_output/'
#clusterStrings = # load data # cluster_text
clusterStrings = cluster_text
'''
Enter your data here
load a list of your clusters.
Each item in the list is a concatenated string of all strings in the cluseter
Example:
    clusterStrings = [
                     ['Text1 in this cluster1. Text2 in this cluster1. Text3 in this cluter1'],
                     ['Text1 in this cluster2. Text2 in this cluster2. Text3 in this cluter2'],
                     ['Text1 in this cluster3. Text2 in this cluster3. Text3 in this cluter3'],
                     ]

'''
#Get centroids of clusters
def get_centroids(cluseterGroups): #clusterGroups -> embeddings
  centroids = []

  for i in range(len(clusterGroups)):
    curCluster = clusterGroups[i]
    centroids.append(np.mean(curCluster, axis=0))
  return centroids
centroids = # load data, 
'''
A list of centroids for each cluster
'''

clusterGroupsIndices = # load data, order of sentences in listofDesc
'''
A 2d list of indices corresponding to the original list of documents.
Each item is a list of indices belonging to the same cluster.
Example:
    clusterGroupsIndices = [
                            [20, 32, 522, 32, 23],
                            [3, 52, 64],
                            [78, 756, 3, 46, 786, 354, 657]
                           ]
'''

listOfDesc = # load data, all the sentences used
'''
A list of documents that were clustered.
Example:
    listOfDesc = [
                 'Sentence in document1. Another sentence in document1.',
                 'Sentence in document2.',
                 'Sentence in document3. Another sentence in document3. Another sentence in document3.'
                 ]
'''
############ End User Input #########################                                    

corpus = clusterStrings



def findWholeWord(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

# distance function Euclidean
def getMyDist(x, y):
  return np.linalg.norm(x - y)

# Word Cloud word sizes are based on TF-IDF of words in cluster
vectorizer = TfidfVectorizer(stop_words=STOPWORDS, norm=None, smooth_idf=False)
X = vectorizer.fit_transform(corpus)

#separate
idf = vectorizer.idf_
# print(idf - 1)
idf = idf - 1
countVectorizer = CountVectorizer(stop_words=STOPWORDS)
tf = countVectorizer.fit_transform(corpus)
tf = tf.toarray()

tfidf = tf*idf

distWeights = {}

lemmatizer = WordNetLemmatizer()
coherenceScores = []
pairWeights = []
temp = []
j = 1
for i in range(len(clusterStrings):
  distWeights = {}
  # df = pd.DataFrame(X[i].T.todense(), index=vectorizer.get_feature_names(), columns=["TF-IDF"])
  df = pd.DataFrame(tfidf[i], index=vectorizer.get_feature_names(), columns=["TF-IDF"])
  dfs = df.sort_values('TF-IDF', ascending=False) 

  df_dict = df.to_dict()
  df_dict1 = df_dict['TF-IDF']
  df_dict1 = {key:val for key, val in df_dict1.items() if val != 0.0}

  # new weighting
  for wcWord in df_dict1:
    for sentenceIdx in range(len(clusterGroupsIndices[i])):
      sentence = listOfDesc[clusterGroupsIndices[i][sentenceIdx]]

      
      if findWholeWord(wcWord)(sentence):

        dist = getMyDist(centroids[i], descEmbeddings[clusterGroupsIndices[i][sentenceIdx]])
      
        if wcWord in distWeights:
          temp = distWeights[wcWord]
          temp.append(dist)
          distWeights[wcWord] = temp
        else:
          distWeights[wcWord] = [dist]
  for wcWord in df_dict1:
    if wcWord not in distWeights:
      distWeights[wcWord] = [1]

  for myWord in distWeights:
    distances = distWeights[myWord]
    sum = 0
    counter = 0
    for mydist in distances:
      # sum = sum + np.exp(-pow(mydist, 2) / (2*0.5))
      sum = sum + 1 / mydist
      counter = counter + 1
    distWeights[myWord] = sum

  for word in df_dict1:
    df_dict1[word] = distWeights[word] * df_dict1[word]
  
  if ' ' in clusterStrings[i]:
    wordcloud = WordCloud(background_color='white', collocations=False).generate_from_frequencies(df_dict1)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(wcFolder + 'cluster' + str(j) + '.png', dpi=600, bbox_inches='tight')
    plt.show()
  j = j + 1
