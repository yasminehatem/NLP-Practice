import nltk
import warnings

warnings.filterwarnings("ignore")
from nltk.corpus import wordnet
#ntlk.download('wordnet)
import random
import io
import string

#data1.txt amazon musical instruments reviews
readFile=io.open('data1.txt','r',errors = 'ignore')
data=readFile.read()
lowerData=data.lower()# converts to lowercase

sentences_tokens = nltk.sent_tokenize(lowerData)# converts to list of sentences
word_tokens = nltk.word_tokenize(lowerData)# converts to list of words


lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

INPUTS = ("hello", "hi", "greetings", "good morning", "good evening","good afternoon",)
RESPONSES = ["hi", "hello",]



#just for greetings
def greeting(sentence):

    for word in sentence.split():
        if word.lower() in INPUTS:
            return random.choice(RESPONSES)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def response(user_response):
    chatterBox_response=''
    sentences_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sentences_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if(req_tfidf==0):
        chatterBox_response=chatterBox_response+"I am sorry! I am not able to get you"
        return chatterBox_response
    else:
        chatterBox_response = chatterBox_response+sentences_tokens[idx]
        return chatterBox_response


flag=True
print("ChatterBox: hey!  My name is ChatterBox. I will answer your queries about ChatterBox. If you want to exit, sat bye or thanks!")

while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("ChatterBox: You are welcome")
        else:
            if(greeting(user_response)!=None):
                print("ChatterBox: "+greeting(user_response))
            else:
                print("ChatterBox: ",end="")
                print(response(user_response))
                sentences_tokens.remove(user_response)
    else:
        flag=False
        print("ChatterBox: Bye! ")
        
        

