# BERT Fine-tuning for Cyber Risk Sentiment Analysis

This document explains the implementation of fine-tuned BERT models for enhanced cyber risk sentiment analysis.

## Overview

The project has been updated to replace the generic BERT classification pipeline with a custom fine-tuning approach specifically designed for cybersecurity domain sentiment analysis.

## Key Improvements

### 1. Domain-Specific Training
- **Before**: Used pre-trained DistilBERT model fine-tuned on general sentiment data (SST-2)
- **After**: Fine-tune BERT specifically on cybersecurity posts and discussions
- **Benefit**: Better understanding of cyber risk vocabulary and context

### 2. Custom Label Support
- **Before**: Limited to positive/negative classification
- **After**: Support for domain-specific sentiment categories (positive, negative, neutral)
- **Benefit**: More nuanced sentiment analysis tailored to cybersecurity discussions

### 3. Improved Accuracy
- **Before**: Generic model may miss cybersecurity-specific sentiment cues
- **After**: Model trained on target domain for improved performance
- **Benefit**: Higher accuracy on cyber risk sentiment classification

## Implementation Details

### Data Preparation
```python
# Filter and prepare cybersecurity data
df_clean = df.dropna(subset=['Cleaned_Text', 'Sentiment_Label']).copy()

# Create label mapping
label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
```

### Model Architecture
- **Base Model**: DistilBERT (lightweight and efficient)
- **Classification Head**: Custom layer for cybersecurity sentiment classes
- **Training**: Fine-tuning with domain-specific data

### Training Process
1. **Data Split**: Train/validation split with stratification
2. **Tokenization**: Text preprocessing with BERT tokenizer
3. **Training Loop**: Custom training with loss optimization
4. **Evaluation**: Performance metrics and validation

## Usage

### Running the Fine-tuning
The fine-tuning implementation is integrated into the main Jupyter notebook:

1. Open `Sentimental_Analysis_model.ipynb`
2. Run the cells in the "BERT for Sentiment Analysis" section
3. The model will automatically:
   - Prepare the data
   - Initialize the model
   - Fine-tune on cybersecurity data
   - Apply predictions to the dataset

### Fallback Mechanism
If model downloading/initialization fails, the implementation includes a fallback to the original pipeline approach, ensuring the analysis can still proceed.

## Technical Specifications

### Dependencies
- `torch`: Deep learning framework
- `transformers`: Hugging Face transformers library
- `datasets`: Data handling and processing
- `scikit-learn`: Evaluation metrics and data splitting

### Hardware Requirements
- **CPU**: Sufficient for small datasets (recommended)
- **GPU**: Optional, will accelerate training if available
- **Memory**: 4GB+ RAM recommended for model loading

### Model Parameters
- **Learning Rate**: 2e-5 (standard for BERT fine-tuning)
- **Batch Size**: 2 (adjusted for small dataset)
- **Epochs**: 2 (limited for demonstration/small dataset)
- **Max Length**: 512 tokens (BERT standard)

## Performance Considerations

### Dataset Size
- Current implementation works with limited data (~26 samples)
- For production use, recommend 1000+ labeled examples per class
- Larger datasets will significantly improve model performance

### Training Time
- Small dataset: 1-5 minutes on CPU
- Larger datasets: May require GPU and longer training time

### Model Storage
- Fine-tuned model: ~250MB
- Includes tokenizer and model weights
- Can be saved and reloaded for future use

## Future Enhancements

1. **Data Augmentation**: Increase training data through text augmentation
2. **Multi-label Classification**: Support for multiple sentiment aspects
3. **Real-time Training**: Continuous learning from new cybersecurity posts
4. **Ensemble Methods**: Combine multiple fine-tuned models
5. **Domain Adaptation**: Transfer learning from related security domains

## Troubleshooting

### Common Issues
1. **Network Access**: Model downloading requires internet connection
2. **Memory Limitations**: Reduce batch size if memory errors occur
3. **Small Dataset**: Consider data augmentation or transfer learning

### Fallback Options
- Pipeline approach automatically used if fine-tuning fails
- Manual label mapping for consistent sentiment categories
- Graceful error handling for missing dependencies

This implementation provides a robust foundation for cybersecurity-specific sentiment analysis while maintaining backward compatibility with the original approach.