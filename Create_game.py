import argparse
import torch
import joblib
import numpy as np
import json
import os
import pandas as pd
import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

def open_file_as_dataframe(file_path_without_extension):
    # Try opening as JSON
    json_path = file_path_without_extension + '.json'
    if os.path.exists(json_path):
        try:
            df = pd.read_json(json_path)
            return df
        except pd.errors.JSONDecodeError:
            pass  # Not a JSON file

    # Try opening as CSV
    csv_path = file_path_without_extension + '.csv'
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            return df
        except pd.errors.EmptyDataError:
            pass  # CSV file is empty

    # If neither JSON nor CSV, return None
    return None


class RNNClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        out = self.softmax(out)
        return out
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        _, predicted = outputs.max(dim=1)
        acc = (predicted == targets).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        _, predicted = outputs.max(dim=1)
        acc = (predicted == targets).float().mean()
        self.log('Vald_loss', loss, prog_bar=True)
        self.log("Vald_acc", acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        _, predicted = outputs.max(dim=1)
        acc = (predicted == targets).float().mean()
        self.log('Test_loss', loss, prog_bar=True)
        self.log("Test_acc", acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

class Data(Dataset):
  def __init__(self, X_train, y_train):
    # need to convert float64 to float32 else 
    # will get the following error
    # RuntimeError: expected scalar type Double but found Float
    self.X = torch.from_numpy(X_train.astype(np.float32))
    # need to convert float64 to Long else 
    # will get the following error
    # RuntimeError: expected scalar type Long but found Float
    self.y = torch.from_numpy(y_train).type(torch.LongTensor)
    self.len = self.X.shape[0]
  
  def __getitem__(self, index):
    return self.X[index], self.y[index]
  def __len__(self):
    return self.len


# Import your model and any necessary preprocessing functions here
#from your_model_module import YourModel
#from preprocessing_module import preprocess_data

def main(test_set_path, model_checkpoint_path, output_json_path, label_enc_path):

    
    # Load the model checkpoint
    #model = RNNClassifier(input_size, hidden_size, num_classes)
    model = torch.load(model_checkpoint_path)
    model.eval()

    # Load the test set
    test_df = open_file_as_dataframe(test_set_path)
    Constructdf = test_df.copy()


     # Load the label encoder (assuming it was saved during training)
    label_encoder = joblib.load(label_enc_path)
    #label_encoder.classes_ = torch.load('label_encoder_classes.pth')  # Change the path

    # Preprocessin Data 
    #test_data = preprocess_data(test_df)  # Modify according to your preprocessing

    # Define the metric to use (mean, median, std)
    med = np.median  # Replace with the desired metric function
    mea = np.mean   # Replace with the desired metric function

    test_df['Mean_norm'] = test_df['norm'].apply(mea)
    test_df['Median_norm'] = test_df['norm'].apply(med)

    test_df['Mean_norm'] = test_df['norm'].apply(mea)
    test_df['Median_norm'] = test_df['norm'].apply(med)

    y = label_encoder.transform(test_df['label'])

    #Drop column that we won't use
    Constructdf = Constructdf.drop('label',axis=1)
    test_df=test_df.drop('label',axis=1)
    test_df=test_df.drop('norm',axis=1)

    #transform Dataframe to Numpy
    X = test_df.to_numpy()

    scaler = StandardScaler()

    scaler.fit(X)
    X_test = scaler.transform(X)
    testdata = Data(X_test, y)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Create DataLoader for batch prediction
    #batch_size = 32  # Adjust according to your needs
    #test_loader = DataLoader(testdata, batch_size=batch_size, num_workers=0, shuffle=False)

    # Make predictions on the test set
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        predictions = model(X_test_tensor)
        _, predicted_classes = predictions.max(dim=1)  # Get predicted classes

    # Convert predictions back to label names
    predicted_labels = label_encoder.inverse_transform(predicted_classes.tolist())

    # Add predicted labels to the test DataFrame
    Constructdf['label'] = predicted_labels

    # Swap the positions of columns 'label' and 'norm'
    Constructdf = Constructdf[['label', 'norm']]

    # Save the DataFrame with predicted labels to a JSON file
    Constructdf.to_json(output_json_path, orient='records')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict labels on a test set using a trained model.")
    parser.add_argument("test_set_path", help="Path to the test set file")
    parser.add_argument("model_checkpoint_path", help="Path to the model file")
    parser.add_argument("output_json_path", help="Path to save the output JSON file")
    parser.add_argument("label_enc_path", help="Path to the lqbel encoder file")
    args = parser.parse_args()

    main(args.test_set_path, args.model_checkpoint_path, args.output_json_path, args.label_enc_path)
