## Data Storytelling: Predicting EU Law Chapters

In this data storytelling, we'll explore the process of predicting the main chapters of EU laws using machine learning techniques. The dataset contains textual information about laws, along with their corresponding directory codes. Our goal is to build a model that can predict the directory codes based on the law text.

### Data Preprocessing and Exploration

We begin by loading the dataset and selecting a subset for demonstration. We preprocess the law text using TF-IDF vectorization to convert the text into numerical features while maintaining the contextual meaning. Additionally, we explore the class distribution to understand the imbalance in the dataset.

### Visualizing Data

A key step in understanding our data is visualizing its distribution. We use a dispersion plot to visualize the class distribution of the dataset. This plot reveals the underrepresented classes that have been oversampled to balance the dataset for better model performance.

### Model Architecture

Our predictive model involves a deep learning architecture. We've chosen the GRU (Gated Recurrent Unit) model due to its ability to capture sequential dependencies in the law text. The model consists of sequential layers of embedding, GRU units, dropout layers for regularization, and a dense layer for multi-label classification.

### Training and Results

We train the GRU model using the preprocessed law text data. The model is trained over several epochs, and we monitor its performance on the validation set. Our evaluation metrics include accuracy, F1 score, precision, and recall. These metrics provide insights into how well the model is performing in predicting the law chapters.

### Insights and Discussion

Our trained model demonstrates promising performance in predicting the law chapters based on the provided text. The high F1 score suggests that the model is effective in capturing both precision and recall.

### Conclusion and Future Work

In this data storytelling, we embarked on a journey to predict EU law chapters using textual data and deep learning techniques. We showcased the preprocessing steps, model architecture, and evaluation metrics. Going forward, we aim to explore techniques that further address class imbalance and fine-tune the model hyperparameters to achieve even better performance.

### Acknowledgments

We extend our gratitude to the ITADATAhack 2023 organizers, our fellow participants, and the institutions collaborating on this event. Special thanks to the libraries and frameworks that powered our analysis and model development.

### References

- [Link to Official Dataset](dataset_link_here)
- [Article on GRU Networks](gru_article_link_here)
- [SMOTE: Synthetic Minority Over-sampling Technique](smote_article_link_here)
- [Keras Documentation](keras_docs_link_here)
