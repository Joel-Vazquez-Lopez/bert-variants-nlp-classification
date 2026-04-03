"""
Imports Section: 
"""
# We need to do:
#  pip3 install transformers
#  pip3 install datasets
# Implementing the model we will use / dataset / also the standard for passing different models
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
from datasets import load_dataset
import torch.nn as nn
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import euclidean_distances, paired_distances, cosine_similarity


# importing the tokenization
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

# For our baseline
from sklearn.dummy import DummyClassifier

#For calculations and creating the table
import pandas as pd
import numpy as np

# For the dataset and dataloader
import torch
from torch.utils.data import Dataset, DataLoader

# For data visualization
import matplotlib.pyplot as plt
import seaborn as sns

results = []

# I needed to investigate how to use and get my GPU for mac active
# Device (Mac GPU if available)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)
torch.set_float32_matmul_precision("medium")


# for the required structure
dataset_films = load_dataset("stanfordnlp/imdb")


# checking the quantity that we have in our datasets
print(len(dataset_films["train"]))
print(len(dataset_films["test"]))


# Here we put the values where we need them
X_train = dataset_films["train"]["text"]
y_train = dataset_films["train"]["label"]
X_test = dataset_films["test"]["text"]
y_test = dataset_films["test"]["label"]


"""
Our Dataset loader we tokenize here: 

"""
class lanextdataset(Dataset):

  def __init__(self,model_name, X, y):

    # We need to do a tokenization of the text this way appears to be the standard
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    # For the other models:
    # self.tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
    # self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    self.X = X
    self.y = y

    """
    following the ideas from: https://huggingface.co/docs/transformers/v4.27.0/preprocessing
    from the tokenization presented by the model
         - we tokenize both our datas,
         - we get the padding done in a simple way to present the same lenghts so there is no problem with the space
         - we truncate to satisfy the model need of space
    """

  def __len__(self):
    return len(self.X)

  def __getitem__(self,idx):

    # we get each sentence of our dataset we tokenize it , we get the padding / truncation and the tensors same with labels
    # i thought i could just put it at the end to run it once, but that was ilogical
    # this place helped me a lot to get the structure proper for what we need for the padding, maybe it was simpler:
    # https://stackoverflow.com/questions/76376455/transformers-tokenizer-attention-mask-for-pytorch

    encoded_x = (self.tokenizer(self.X[idx], max_length = 128, truncation=True, padding="max_length"))
    encoded_y = self.y[idx]

    # originally i used (return_tensors="pt") within each tokenization of each review, that was problematic, as it added one extra
    # dimention for each review, such as [1,128], and i needed to restructure that as it gave me an error in the model
    # Then i needed to follow: https://docs.pytorch.org/docs/stable/tensors.html to actually do the proper way tensors, without

    words = torch.tensor(encoded_x["input_ids"])
    attention_mask =  torch.tensor(encoded_x["attention_mask"])
    labels = torch.tensor(encoded_y)


    return words, attention_mask, labels
  


"""
Our Sentiment classifier: 

"""

class Sentiment(torch.nn.Module):

  """
  in order to initialize our model i had to check for what to pass in our constructor, we need our label structure of the model:
  these places helped deeply:
  https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/
  https://medium.com/%40rajratangulab.more/fine-tuning-bert-for-text-classification-using-hugging-face-transformers-685c132d185d

  """

  def __init__(self,model_name, n_labels = 2, dropout = 0.3, two_layers= False):
    super(Sentiment, self).__init__()

    self.model = AutoModel.from_pretrained(model_name)

    # For the dropout in case needed
    self.dropout = torch.nn.Dropout(dropout)

    """

    I was not completely sure what the model hidden size is, i saw a lot of: 768, but better to get it from the model itself
    Here we do our layer:
    - self.layer = nn.Linear(self.bert.config.hidden_size, n_labels)
    however i wanted to try to do a mulitlayer model to experiment more with that,
    so better use the number that bert accept

    """
    # to have a gate to pass one or two options for the layers.
    self.two_layers = two_layers
    
    # One layer
    self.oglayer = nn.Linear(self.model.config.hidden_size, n_labels)
    
    # Two layers
    self.layer = nn.Linear(self.model.config.hidden_size, 256)
    # As we discussed for the layers to have dimensionality, and there is non linearity we applied relu.
    self.relu = nn.ReLU()

    # I want to experiment to see how it works having more than one layer, and what things we can do with it, as we discussed in the
    # paper itself about Finetuning bert
    self.layer2 = nn.Linear(256, n_labels)


  # Here we pass the variables we created in our dataset / dataloader
  def forward(self, words, attention_mask):

    # we do this to ignore the vectors of the words, as we are only interested on the vector of the whole sentence
    # we pass the vectors of the words and if is word or padding
    # _, pooled_output = self.model(input_ids=words, attention_mask=attention_mask)

    outputs = self.model(input_ids=words, attention_mask=attention_mask)

    try:
      # we ignore the first output, we get the second one the same as _,pooled_output =
      pooled_output = outputs[1]

    except:
      # Here we build the pooled output ourselves as destilbert doesnt provide one
      # like Bert, where we extract the first token which is our CLS
      hidden = outputs.last_hidden_state
      # hidden has shape (batch_size, sequence_length, hidden_dim) we need a vector per sentence
      # We take:
      # all sentences in the batch (:)
      # the first token (cls)
      # all hidden features (:)
      pooled_output = hidden[:, 0, :]

    # we apply the dropout, for the two layer system and one layer system
    output = self.dropout(pooled_output)

    if self.two_layers:
        output = self.layer(output)
        # and then we need to pass it here for it to be used. 
        output = self.relu(output)
        output = self.layer2(output)
    else:
        # we use the first layer result
        output = self.oglayer(output)

    # we return it
    # for testing only with one layer / or both layers 
    return output
    



  def fit(self, dataloader):
    # we tell the model that we are doing the training process
    self.train()

    # Because of computing power we will keep it small
    epochs = 3

    # here we compute the loss
    loss_function = torch.nn.CrossEntropyLoss().to(device)

    """
    In case that we want to use another loss, we need to apply the results the
    softmax However based on what ive seen with crossentropy
    (to get the log-probabilities) it already applies it, in case that is needed:

    loss_function = nn.NLLLoss(ignore_index=tag2idx['<PAD>'])
    output.softmax(model(self.words, self.attention_mask), dim = 1)
    """

    # this is our SGD with Adam
    optimizer = torch.optim.AdamW(self.parameters(), lr=2e-5)

    # This to have less data and only averaging the loss and accuracy per epoch
    total_loss = []
    total_accuracy = []

    # I want to check what would happen if we append all the batches for the visualization
    experiment_loss = []
    experiment_accuracy = []

    # we loop a total of epoch times
    for epoch in range(epochs):

      # Our lists for the experiment i want to do
      loss_per_batch = []
      accuracy_per_batch = []

      correct = 0
      total = 0

      # here we get them for visualization /// also works as our loop to
      # not looping over the data on n2

      with tqdm(dataloader, total=len(dataloader), unit="batch") as batch:

        # I needed to add this loop as we did in other works to access the data
        for i, data in enumerate(batch):

          # we get each variable of our dataloader separated
          words = data[0].to(device)
          attention_mask = data[1].to(device)
          labels = data[2].to(device)

          optimizer.zero_grad()
          # we call our forward with this variables
          scores = self(words, attention_mask)

          # we compute the loss of the previous line
          loss = loss_function(scores,labels)

          # this is our backpropagation
          loss.backward()

          # the steps for our SGD
          optimizer.step()

          # We get the accuracy
          accuracy = torch.mean((torch.argmax(scores, dim=1) == labels).float())

          # for the epoch to show more information
          batch.set_postfix(loss=loss.item(), accuracy=accuracy)

          accuracy = accuracy.item()

          # just to get the actual accuracy not per batch
          pred = torch.argmax(scores, dim=1)
          correct += (pred == labels).sum().item()
          total += labels.size(0)

          # we put loss / accuracy in a clean list
          loss_per_batch.append(loss.item())
          accuracy_per_batch.append(accuracy)

          experiment_loss.append(loss.item())
          experiment_accuracy.append(accuracy)

        total_loss.append(sum(loss_per_batch)/len(loss_per_batch))
        total_accuracy.append((correct/total)*100)
    self.total_loss = total_loss
    self.total_accuracy = total_accuracy
    self.experiment_loss = experiment_loss
    self.experiment_accuracy = experiment_accuracy
    # to get the training accuracy 
    return correct/total*100


  def score(self,dataloader_test):
    # here we tell the model that we are in the evaluation part, this is for a
    # safe structure (some people do it but is not completely needed)
    self.eval()

    n_correct = 0
    n_total = 0

    with torch.no_grad():

      for data in dataloader_test:

        words = data[0].to(device)
        attention_mask = data[1].to(device)
        labels = data[2].to(device)

        scores = self(words, attention_mask)

        predictions = scores.argmax(dim=1)

        n_correct += (predictions == labels).sum().item()
        # here we have the info of how many examples we have, it took me a long time this
        n_total += len(labels)

      accuracy = 100 * n_correct / n_total

      return accuracy


"""
First Model Bert: 

"""

model_name = "bert-base-uncased"

dataset = lanextdataset(model_name, X_train, y_train)
dataset_test = lanextdataset(model_name, X_test, y_test)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=8)

modelBert = Sentiment(model_name, two_layers=False)
modelBert = modelBert.to(device)
train = modelBert.fit(dataloader)

results.append({"Model": "Bert",
                "Dataset": "Train",
               "Accuracy": train,
                })

score = modelBert.score(dataloader_test)

results.append({"Model": "Bert",
                "Dataset": "Test",
               "Accuracy": score,
                })

print(f"Bert Train Accuracy: {train:.2f}%, \nBert Test Accuracy: {score:.2f}%")

# to save the values for visualization.
bert1_loss = modelBert.total_loss
bert1_acc = modelBert.total_accuracy

# To free up the space so they are not stcking 
del modelBert
torch.cuda.empty_cache()

modelBert2 = Sentiment(model_name, two_layers=True)
modelBert2 = modelBert2.to(device)
train = modelBert2.fit(dataloader)

results.append({"Model": "Bert-Two Layers",
                "Dataset": "Train",
               "Accuracy": train,
                })

score = modelBert2.score(dataloader_test)
results.append({"Model": "Bert-Two Layers",
                "Dataset": "Test",
               "Accuracy": score,
                })


print(f"Bert-Two Layers Train Accuracy: {train:.2f}%, \nBert-Two Layers Test Accuracy: {score:.2f}%")

bert2_loss = modelBert2.total_loss
bert2_acc = modelBert2.total_accuracy

# To free up the space so they are not stcking 
del modelBert2
torch.cuda.empty_cache()

"""
Second Model DestilBert: 

"""

model_name = "distilbert-base-uncased"

dataset = lanextdataset(model_name, X_train, y_train)
dataset_test = lanextdataset(model_name, X_test, y_test)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=8)

modelDestilBert = Sentiment('distilbert-base-uncased')
modelDestilBert = modelDestilBert.to(device)
train = modelDestilBert.fit(dataloader)

results.append({"Model": "DestilBert",
                "Dataset": "Train",
               "Accuracy": train,
                })

score = modelDestilBert.score(dataloader_test)

results.append({"Model": "DestilBert",
                "Dataset": "Test",
               "Accuracy": score,
                })

print(f"DestilBert Train Accuracy: {train:.2f}%, \nDestilBert Test Accuracy: {score:.2f}%")

destil_loss = modelDestilBert.total_loss
destil_acc = modelDestilBert.total_accuracy

del modelDestilBert
torch.cuda.empty_cache()

"""
Third Model Multilingual Bert: 

"""

model_name = "bert-base-multilingual-cased"

dataset = lanextdataset(model_name, X_train, y_train)
dataset_test = lanextdataset(model_name, X_test, y_test)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=8)

modelMULTBert = Sentiment('bert-base-multilingual-cased')
modelMULTBert = modelMULTBert.to(device)
train = modelMULTBert.fit(dataloader)

results.append({"Model": "Multilingual Bert",
                "Dataset": "Train",
               "Accuracy": train,
                })

score = modelMULTBert.score(dataloader_test)

results.append({"Model": "Multilingual Bert",
                "Dataset": "Test",
               "Accuracy": score,
                })

print(f"Multilingual Bert Train Accuracy: {train:.2f}%, \nMultilingual Bert Test Accuracy: {score:.2f}%")


multi_loss = modelMULTBert.total_loss
multi_acc = modelMULTBert.total_accuracy

del modelMULTBert
torch.cuda.empty_cache()


"""
Table of results:
"""

results_table = pd.DataFrame(results)
final_table = results_table.pivot_table(
    values="Accuracy",
    index="Model",
    columns="Dataset"
)
print("\nFinal Results Table:")
print(final_table)


bert_vs_layers = results_table[
    results_table["Model"].isin(["Bert", "Bert-Two Layers"])
].pivot_table(
    values="Accuracy",
    index="Model",
    columns="Dataset"
)
print("\nBert vs Bert2layers Results:")
print(bert_vs_layers)


bert_vs_distil = results_table[
    results_table["Model"].isin(["Bert", "DestilBert"])].pivot_table(values="Accuracy",
                                                                     index="Model",
                                                                     columns="Dataset"
                                                                     )
print("\nBert vs DestilBert Results:")
print(bert_vs_distil)


bert_vs_multi = results_table[
    results_table["Model"].isin(["Bert", "Multilingual Bert"])].pivot_table(values="Accuracy",
                                                                            index="Model",
                                                                            columns="Dataset"
                                                                              )
print("\nBert vs Multilingual Bert Results:")
print(bert_vs_multi)


final_table.to_csv("final_results.csv")


"""
Visualization of the results:
"""


"""
LOSS PLOT
"""

plt.figure()

plt.plot(bert1_loss, marker='o', label="BERT (1 layer)")
plt.plot(bert2_loss, marker='o', label="BERT (2 layers)")
plt.plot(destil_loss, marker='o', label="DistilBERT")
plt.plot(multi_loss, marker='o', label="Multilingual BERT")

plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("loss_per_epoch.png")
plt.show()
print("Loss per epoch plot saved as 'loss_per_epoch.png'")

"""
ACCURACY PLOT
"""

plt.figure()

plt.plot(bert1_acc, marker='o', label="BERT (1 layer)")
plt.plot(bert2_acc, marker='o', label="BERT (2 layers)")
plt.plot(destil_acc, marker='o', label="DistilBERT")
plt.plot(multi_acc, marker='o', label="Multilingual BERT")

plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.savefig("accuracy_per_epoch.png")
plt.show()

print("Accuracy per epoch plot saved as 'accuracy_per_epoch.png'")