# main.py
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader

# Load data
def load_data(file_paths):
    resumes = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            resume = f.read()
            resumes.append(resume)
    return resumes

file_paths = ['data/first_question1.txt', 'data/first_question2.txt',
              'data/first_question3.txt', 'data/first_question6.txt',
              'data/first_question7.txt', 'data/first_question8.txt',
              'data/first_question9.txt', 'data/first_question10.txt',
              'data/first_question11.txt', 'data/first_question12.txt',
              'data/first_question13.txt', 'data/first_question14.txt',
              'data/first_question15.txt']

resumes = load_data(file_paths)

# Tokenize data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_resumes = []
for resume in resumes:
    tokenized_resume = tokenizer.encode_plus(
        resume,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    tokenized_resumes.append(tokenized_resume)

# Create dataset class
class BertDataset(Dataset):
    def __init__(self, tokenized_resumes, labels):
        self.tokenized_resumes = tokenized_resumes
        self.labels = labels

    def __len__(self):
        return len(self.tokenized_resumes)

    def __getitem__(self, idx):
        tokenized_resume = self.tokenized_resumes[idx]
        label = self.labels[idx]
        return {
            'input_ids': tokenized_resume['input_ids'].flatten(),
            'attention_mask': tokenized_resume['attention_mask'].flatten(),
            'labels': torch.tensor(label)
        }

# Create dataset and data loader
labels = [0.5, 0.8, 1, 1, 0.3, 0.1, 0.9, 0.3, 1, 0.1, 0.7, 1, 0.5]

dataset = BertDataset(tokenized_resumes, labels)
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define model and optimizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
optimizer = AdamW(model.parameters(), lr=1e-5)

# Train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = torch.nn.MSELoss()(outputs.logits.squeeze(), labels.squeeze())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}')

    model.eval()
# Save trained model
torch.save(model.state_dict(), 'trained_model.pt')