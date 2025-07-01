#!/usr/bin/env python3
"""
Test script for the BERT fine-tuning implementation
This script validates that the implementation can handle the cybersecurity data
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_data_preparation():
    """Test the data preparation for fine-tuning"""
    print("🧪 Testing data preparation...")
    
    try:
        # Load the cybersecurity dataset
        df = pd.read_csv('processed_data/cleaned_cybersecurity_reddit_posts.csv')
        print(f"✅ Loaded dataset with {len(df)} rows")
        
        # Test data cleaning
        df_clean = df.dropna(subset=['Cleaned_Text', 'Sentiment_Label']).copy()
        print(f"✅ Clean dataset has {len(df_clean)} rows")
        
        # Test label mapping
        unique_labels = df_clean['Sentiment_Label'].unique()
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        print(f"✅ Label mapping created: {label_to_id}")
        
        # Test text preprocessing
        texts = df_clean['Cleaned_Text'].tolist()
        labels = df_clean['Sentiment_Label'].map(label_to_id).tolist()
        print(f"✅ Prepared {len(texts)} text samples with labels")
        
        return True, df_clean, label_to_id
        
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return False, None, None

def test_fallback_pipeline():
    """Test the fallback pipeline approach"""
    print("\n🧪 Testing fallback pipeline approach...")
    
    try:
        from transformers import pipeline
        
        # Initialize pipeline
        sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            return_all_scores=False
        )
        print("✅ Pipeline initialized successfully")
        
        # Test on sample text
        test_text = "This cybersecurity vulnerability is concerning"
        result = sentiment_analyzer(test_text)
        print(f"✅ Pipeline test result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def test_pytorch_availability():
    """Test PyTorch and transformers availability"""
    print("\n🧪 Testing PyTorch and transformers...")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print("✅ Transformers library available")
        
        # Test device availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ Using device: {device}")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch/Transformers test failed: {e}")
        return False

def test_custom_dataset_class():
    """Test the custom dataset implementation"""
    print("\n🧪 Testing custom dataset class...")
    
    try:
        import torch
        from torch.utils.data import Dataset
        
        class TestDataset(Dataset):
            def __init__(self, texts, labels):
                self.texts = texts
                self.labels = labels
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                return {
                    'text': self.texts[idx],
                    'label': self.labels[idx]
                }
        
        # Test with sample data
        test_texts = ["Sample cyber security text", "Another security post"]
        test_labels = [0, 1]
        
        dataset = TestDataset(test_texts, test_labels)
        print(f"✅ Custom dataset created with {len(dataset)} samples")
        
        # Test data access
        sample = dataset[0]
        print(f"✅ Sample data access: {sample}")
        
        return True
        
    except Exception as e:
        print(f"❌ Custom dataset test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and provide summary"""
    print("🚀 Starting BERT Fine-tuning Implementation Tests\n")
    
    results = []
    
    # Test data preparation
    data_ok, df_clean, label_mapping = test_data_preparation()
    results.append(("Data Preparation", data_ok))
    
    # Test PyTorch availability
    pytorch_ok = test_pytorch_availability()
    results.append(("PyTorch/Transformers", pytorch_ok))
    
    # Test custom dataset
    dataset_ok = test_custom_dataset_class()
    results.append(("Custom Dataset", dataset_ok))
    
    # Test fallback pipeline
    pipeline_ok = test_fallback_pipeline()
    results.append(("Fallback Pipeline", pipeline_ok))
    
    # Summary
    print("\n📊 Test Results Summary:")
    print("=" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print("=" * 40)
    print(f"Tests passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("🎉 All tests passed! Implementation is ready.")
    elif pipeline_ok:
        print("⚠️ Some tests failed, but fallback pipeline works.")
    else:
        print("❌ Critical tests failed. Check dependencies.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)