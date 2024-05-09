
import torch

with open('data/first_question1.txt', 'r', encoding='utf-8') as f:
    resume1 = f.read()

with open('data/first_question2.txt', 'r', encoding='utf-8') as f:
    resume2 = f.read()

with open('data/first_question3.txt', 'r', encoding='utf-8') as f:
    resume3 = f.read()


#
# print(resume1)
# print()
# print(resume2)
# print()
# print(resume3)

labels = [0.5, 0.8, 1]

from transformers import BertTokenizer

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the resumes
tokenized_resume1 = tokenizer.encode(resume1, add_special_tokens=True)
tokenized_resume2 = tokenizer.encode(resume2, add_special_tokens=True)
tokenized_resume3 = tokenizer.encode(resume3, add_special_tokens=True)

tokenized_resumes = [tokenized_resume1, tokenized_resume2, tokenized_resume3]
# labels = [0.5, 0.8, 1]


input_examples = []
for input_ids, label in zip(tokenized_resumes, labels):
    attention_mask = [1] * len(input_ids)
    label = torch.tensor(label)
    input_example = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': label
    }
    input_examples.append(input_example)

# Print the input examples
for input_example in input_examples:
    print(input_example)





from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, AdamW

# Create a custom dataset class
class BertDataset(Dataset):
    def __init__(self, input_examples):
        self.input_examples = input_examples

    def __len__(self):
        return len(self.input_examples)

    def __getitem__(self, idx):
        input_example = self.input_examples[idx]
        input_ids = input_example['input_ids']
        attention_mask = [1] * len(input_ids)
        padding_length = 512 - len(input_ids)
        if padding_length > 0:
            attention_mask += [0] * padding_length
        input_ids += [0] * padding_length

        label = input_example['labels']
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'labels': label.unsqueeze(0)
        }

# Create an instance of the custom dataset class
dataset = BertDataset(input_examples)

# Create a data loader from the dataset
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define a BERT-based model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', output_attentions=False, output_hidden_states=False)

# Define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-5, no_deprecation_warning=True)
# loss_fn = torch.nn.MSELoss()
loss_fn = torch.nn.BCEWithLogitsLoss()

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(5):  # Train for 5 epochs
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        # loss = loss_fn(outputs, labels)

        outputs = model(input_ids, attention_mask=attention_mask)

        # remove the extra dimension from the output tensor
        loss = loss_fn(outputs.logits.squeeze(),  labels.squeeze())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}')

    model.eval()
# save the trained mode
torch.save(model.state_dict(), 'trained_model.pt')

torch.save(model.classifier.weight, 'classifier_weight.pt')
torch.save(model.classifier.bias, 'classifier_bias.pt')