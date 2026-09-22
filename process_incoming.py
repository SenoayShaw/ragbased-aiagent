import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests


def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    embedding = r.json()["embeddings"]
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
            #"model": "deepseek-r1:latest",
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        })
    
    response = r.json()
    #print(response) #it will print the entire response from the API, which includes the generated text and other metadata.
    return response
    



df= joblib.load('embeddings.joblib')




incoming_query=input("Ask me a question about the data: ")
query_embedding = create_embedding([incoming_query])[0]


#print(np.vstack(df['embedding'].values))
#print(np.vstack(df['embedding']).shape)

#find similarities of query_embedding with other embeddings
similarities = cosine_similarity(np.vstack(df['embedding']),[query_embedding]).flatten()
#print(similarities)
top_results=5
max_indx=similarities.argsort()[::-1][0:top_results]# print the indices of the top 3 most similar chunks
#print(max_indx)
new_df=df.iloc[max_indx]
#print(new_df[['title','number','text']])

prompt=f'''I am teaching web development in my Sigma webdevelopment course.
Here are video subtitle chunks containing video title,video number,start time in
seconds,end time in seconds,text at that time:

{new_df[['title','number', 'start', 'end', 'text']].to_json(orient='records')}
-------------------


"{incoming_query}"
User asked this question related to the video chunks,you have to answer in a human way (dont mention the above format,its just for your reference) where and how much 
content is taught  (in which video and what timestamp)and guide the used to go to that particular video.
if the user asks unrelated question,tell him that you can only answer questions related to the course.

'''

with open('prompt.txt','w') as f:
    f.write(prompt)
    
response = inference(prompt)["response"]
print(response)

with open('response.txt','w') as f:
    f.write(response)

#for index,item in new_df.iterrows():
   # print(index,item['text'],item['title'],item['number'],item['start'],item['end'])