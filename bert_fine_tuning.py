"""
BERT Fine-tuning for Cyber Risk Sentiment Analysis
This module replaces the pre-trained pipeline approach with actual fine-tuning
of BERT on the cybersecurity domain data.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

class CyberRiskDataset(Dataset):
    """Dataset class for cyber risk sentiment analysis"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class CyberRiskBERTFineTuner:
    """Main class for fine-tuning BERT on cyber risk data"""
    
    def __init__(self, model_name='distilbert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.label_mapping = {}
        self.reverse_label_mapping = {}
        
    def prepare_data(self, df, text_column='Cleaned_Text', label_column='Sentiment_Label'):
        """Prepare data for training"""
        # Filter out NaN values
        df_clean = df.dropna(subset=[text_column, label_column]).copy()
        
        # Create label mapping
        unique_labels = df_clean[label_column].unique()
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.reverse_label_mapping = {idx: label for label, idx in self.label_mapping.items()}
        
        print(f"Label mapping: {self.label_mapping}")
        
        # Convert labels to integers
        df_clean['label_ids'] = df_clean[label_column].map(self.label_mapping)
        
        # Split data
        texts = df_clean[text_column].tolist()
        labels = df_clean['label_ids'].tolist()
        
        # Use stratified split to maintain label distribution
        try:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels
            )
        except ValueError:
            # If stratification fails due to small dataset, use regular split
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=0.2, random_state=42
            )
        
        return train_texts, val_texts, train_labels, val_labels
    
    def initialize_model(self, num_labels):
        """Initialize tokenizer and model"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=num_labels
        )
        
    def create_datasets(self, train_texts, val_texts, train_labels, val_labels):
        """Create PyTorch datasets"""
        train_dataset = CyberRiskDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = CyberRiskDataset(val_texts, val_labels, self.tokenizer)
        
        return train_dataset, val_dataset
    
    def setup_training(self, train_dataset, val_dataset, output_dir='./cyber_risk_bert'):
        """Setup training arguments and trainer"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            report_to=None,  # Disable wandb/tensorboard logging
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
    
    def train(self):
        """Train the model"""
        print("Starting training...")
        self.trainer.train()
        print("Training completed!")
        
        # Evaluate on validation set
        eval_results = self.trainer.evaluate()
        print(f"Validation Results: {eval_results}")
        
        return eval_results
    
    def save_model(self, save_directory='./cyber_risk_bert_final'):
        """Save the trained model"""
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        
        # Save label mappings
        import json
        with open(f"{save_directory}/label_mapping.json", 'w') as f:
            json.dump(self.label_mapping, f)
        
        print(f"Model saved to {save_directory}")
    
    def predict(self, texts, batch_size=8):
        """Make predictions on new texts"""
        self.model.eval()
        predictions = []
        confidences = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                batch_predictions = torch.argmax(probs, dim=-1)
                batch_confidences = torch.max(probs, dim=-1)[0]
                
                predictions.extend(batch_predictions.cpu().numpy())
                confidences.extend(batch_confidences.cpu().numpy())
        
        # Convert predictions back to labels
        predicted_labels = [self.reverse_label_mapping[pred] for pred in predictions]
        
        return predicted_labels, confidences

# Main execution function
def main():
    """Main function to run the fine-tuning process"""
    print("Loading data...")
    df = pd.read_csv('processed_data/cleaned_cybersecurity_reddit_posts.csv')
    
    # Initialize fine-tuner
    fine_tuner = CyberRiskBERTFineTuner()
    
    # Prepare data
    train_texts, val_texts, train_labels, val_labels = fine_tuner.prepare_data(df)
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    
    # Initialize model
    num_labels = len(fine_tuner.label_mapping)
    fine_tuner.initialize_model(num_labels)
    
    # Create datasets
    train_dataset, val_dataset = fine_tuner.create_datasets(
        train_texts, val_texts, train_labels, val_labels
    )
    
    # Setup training
    fine_tuner.setup_training(train_dataset, val_dataset)
    
    # Train
    eval_results = fine_tuner.train()
    
    # Save model
    fine_tuner.save_model()
    
    return fine_tuner, eval_results

if __name__ == "__main__":
    fine_tuner, results = main()
    print("Fine-tuning completed successfully!")
    print(f"Final results: {results}")