
#%%
#1. Import libraries
from modules import text_cleaning, lstm_model_creation
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences, plot_model
import datetime
import json
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas as pd
import pickle

%matplotlib inline

#%%
#Set parameters
num_words = 300 # unique number of words in all the sentences
oov_token = '<OOV>' # out of vocabulary
SEED = 42
BATCH_SIZE = 64
EPOCHS = 5
embedding_layer = 64
filename = 'True.csv'
nltk.download('stopwords')
stop_words = stopwords.words('english')
replace_text = r"\bSource link\b"

#%%
# 2. Data Loading
CSV_PATH = os.path.join(os.getcwd(), 'dataset', filename)
df = pd.read_csv(CSV_PATH)

#%%
# 3. Data Inspection
df.describe()
df.info()
df.head()

#%%
#206 duplicated data here
df.duplicated().sum()

#%%
# To check NaN
df.isna().sum()

#%%
print(df['text'][0])
#temp = df(['text'][0])

#%%
#4. Data Cleaning

for index, temp in enumerate(df['text']):
    df['text'][index] = text_cleaning(temp)
 
    
#df['text'][index] = re.sub('[^a-zA-Z]', ' ', temp).lower()


# use pythex.org

#%%
#5. Features Selection
text = df['text']
subject = df['subject']

# Filter the text using stopwords
text = [word for word in text if word.lower() not in stop_words]

#%%
#6. Data Preprocessing

# Tokenizer
tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token) # instantiate

tokenizer.fit_on_texts(text)
#to transform the text using tokenizer --> mms.transform
text = tokenizer.texts_to_sequences(text)

#%%
# Padding
text = pad_sequences(text,maxlen=300, padding='post', truncating='post')

#%%
# One hot encoder
ohe = OneHotEncoder(sparse=False)
subject = ohe.fit_transform(subject[::, None])

#%%
# Train test split
# expand dimension before feeding to train_test_split
#padded_text = np.expand_dims(padded_text, axis=-1)

X_train,x_test,y_train,y_test = train_test_split(text, subject, shuffle=True, test_size=0.2, random_state=SEED)

#%%
#7. Model Development
model = lstm_model_creation(num_words, subject.shape[1])

#%%
#Model Training
es_callback = EarlyStopping(monitor='val_loss',patience=5,verbose=0,restore_best_weights=True)

hist = model.fit(X_train,y_train,validation_data=(x_test,y_test),batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[es_callback])
#%%
#8. Model Analysis
hist.history.keys()

#%%

plt.figure()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['training', 'validation'])
plt.show()

#%%
y_predicted = model.predict(x_test)
y_predicted = np.argmax(y_predicted,axis=1)
y_test = np.argmax(y_test,axis=1)

#%%
print(classification_report(y_test, y_predicted))
cm = (confusion_matrix(y_test, y_predicted))

#%%
disp = ConfusionMatrixDisplay(cm)
disp.plot()

#%%
#9. Model Saving

# to save trained model
model.save('model.h5')

# to save one hot encorder model
with open('ohe.pkl','wb') as f:
    pickle.dump(ohe, f)

# %%
# to save tokenizer

token_json = tokenizer.to_json()
with open('tokenizer.json','w') as f:
    #json.dump(token_json,f)
    json.dump(tokenizer.to_json(),f)

# %%
