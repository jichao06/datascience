from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from numpy import linalg
from nltk.stem import SnowballStemmer
import numpy as np

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = TfidfVectorizer.build_analyzer(self)
        english_stemmer = SnowballStemmer('english')
        return lambda doc:(english_stemmer.stem(w) for w in analyzer(doc))
        
class DocumentClusterizer():
    def __init__(self):
        self.vectorizer = StemmedTfidfVectorizer()
        self.km = KMeans(50, init='random', n_init=1, verbose=1)
    
    def train(self, data):
        self.data = data
        self.vectorized = self.vectorizer.fit_transform(data)
        self.km.fit(self.vectorized)
        
    def find_most_similar(self, example):
        print 'EXAMPLE:', example
        example = self.vectorizer.transform([example])
        pred = self.km.predict(example)[0]
        similar_indices = (self.km.labels_==pred).nonzero()[0]
        similar = []
        for i in similar_indices:
            dist = linalg.norm((self.vectorized[i]-example).toarray())
            similar.append((dist, self.data[i]))
        similar = sorted(similar)
        print 'SIMILAR:', similar[0]
            
        
        
def exercise():
    groups = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 
        'comp.sys.mac.hardware', 'comp.windows.x', 'sci.space']
    train_data = fetch_20newsgroups(subset='train', categories=groups)
    clusterizer = DocumentClusterizer()
    clusterizer.train(train_data.data)
    test_data = fetch_20newsgroups(subset='test', categories=groups)
    for i in range(10):
        sample = test_data.data[np.random.randint(len(test_data.data))]
        clusterizer.find_most_similar(sample)
    
    