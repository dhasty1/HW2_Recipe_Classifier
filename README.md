# HW2: Recipe Classifier

### Group Members:
- Logan Hasty
- Devin Horst
- Jacob Watson
- Nish Chandan

## Recipe Classifier Write-Up

Analyzing our inter-annotator agreement shows that 4 annotators labeled a total of 797 records using two different categories/classes. We can see that we were successful in ensuring that we all annotated the same records by looking at the “Co-Incident Examples” value of 200; this indicates that there were 200 examples that were identically labeled by multiple annotators. This is a great indicator as to the potential validity of our answers, as the fact that at least two annotators labeled every single record identically suggests that the annotations we are feeding the model are likely accurate and will not contradict each other, as this could cause the model to generate inaccurate predictions otherwise. 

Examining our Agreement Statistics further supports this conclusion, as our Simple Agreement score of 0.8317 shows that we all assigned the same labels to approximately 83% of the records from our evaluation dataset. While this metric is encouraging, it is important to remember that there were far more records that were not relevant when compared to those that were relevant, so the Simple Agreement score is a bit misleading since our classes were not balanced. Therefore, another metric that might provide a better understanding of the agreement among our annotators is Krippendorff's Alpha score.

Our Krippendorff's Alpha score provides insight into the level of agreement between annotators/annotations, looking beyond what would be expected by pure chance or traditional probabilities. Our Krippendorff's Alpha score of 0.5889 demonstrates that there was a moderate amount of agreement among our annotators, which would be sufficient for the purposes of this model since an incorrect label/classification most likely would not have any severe consequences outside the degradation of the model’s capabilities or predictive power. However, this score does show that there is room for improvement when it comes to agreement among annotators; it also supports the importance of the process of developing a “gold standard” dataset, as this will contribute to an accurate classifier by ensuring that the model is ingesting accurate and consistent information during the training process. 

Lastly, we considered our Gwet's AC2 score, as this metric is more sensitive to differing annotations, and is particularly sensitive in our case since we have a much higher number of non-relevant records. To account for this class imbalance, the Gwet’s AC2 metric adjusts the typically expected agreement ratios by placing a weight on each class to ensure that the importance of each annotation is properly recognized. This will make sure that small changes in the agreement patterns of our “relevant” class will be more highly scrutinized and more influential to the model/training process, even though this class is much less prevalent in our dataset. In summary, our Gwet's AC2 score of 0.7155 further indicates that there was a sufficient level of agreement between our annotators, although it does still suggest that there is room for improvement in the model. 

In terms of modeling methodology, we first had to remove duplicate annotations to be sure our annotated dataset could be properly imported and help train the model. We accomplished this with a Python script that queried the json file and removed any entries that were annotated by a particular annotation two times, as these entries appeared twice in our dataset. Next, we imported the newly cleaned dataset into our VS Code environment so it could be used to train our Prodigy model; we also utilized this clean dataset to develop our “gold” dataset, which we compiled by reviewing our annotations as a group and deciding which of them were most accurate. Once we determined how each record would be best classified, we combined these ideal classifications so the model would be able to train on a dataset that we knew was accurate and consistent. This information is essential to the training process, as we then utilized this final dataset to train the classifier by comparing the labeled training data against the “gold” dataset we developed earlier in the model-building process. Since we know that the model’s initial classifications are being compared to our dataset which contains the ideal label for each record, we can assess the performance of the classifier based on whether the model’s training prediction matches the “gold” standard we established in our final dataset.

To further evaluate the performance of our Prodigy model, we thought it would be beneficial to generate a second cooking-related text classification model using spaCy, as this would allow us to compare overall performance as well as the specific strengths and weaknesses of each technique. We started by formatting our cleaned dataset so it could be used for training/analysis in our spaCy model; since spaCy utilizes a binary format, we had to reformat our json dataset to ensure that it would be accepted by spaCy’s training pipeline. Once this was completed, we used the newly formatted dataset to initiate training a custom Natural Language Processing model with spaCy, and then generated a summary of the classifier’s performance so we could compare the spaCy model to our initial Prodigy-based classifier to see which performed better, as well as analyze strengths and weaknesses of each model.

Our prodigy model (metrics below) exhibits promising performance as seen by decreasing loss values and improving textcat and cats scores over training iterations, indicating effective learning and enhanced text categorization capabilities. Furthermore, the model achieves commendable F1-scores for both "RELEVANT" and "NOT_RELEVANT" labels, with precision, recall, and F1-score values ranging from 56.67% to 85.71% and 59.65% to 83.92%, respectively. These metrics demonstrate the model's ability to accurately classify texts into relevant and non-relevant categories, with a balanced trade-off between precision and recall. Additionally, the ROC AUC values of 0.75 for both labels suggest satisfactory discrimination ability between classes. Overall, these results highlight the effectiveness of the trained model in text classification tasks, showcasing its potential for practical applications in information retrieval and content filtering.

On the other hand, our spacy model (metrics below) exhibits promising trends, showcasing its efficacy in text classification tasks. Throughout the training iterations, the model demonstrates consistent improvement in categorization accuracy, as evidenced by the steady increase in the Cats Score metric. Concurrently, the loss values for both the Textcat and Tok2Vec components remain low, indicating efficient learning and word vectorization processes. Importantly, the model achieves a commendable Cats Score of up to 69.43%, reflecting its ability to accurately classify texts into predefined categories. Moreover, the training process maintains a consistent speed, ensuring efficient processing of large datasets without significant variations in training time. Overall, these results underscore the spaCy model's robust performance and potential for practical applications in automated text analysis and categorization across diverse domains.

Comparing both the models, the spaCy model achieved a higher Cats Score of up to 69.43%, indicating a better ability to accurately classify texts into predefined categories compared to the Prodigy model, which achieved a Cats Score of 68.60%. Additionally, the spaCy model exhibited consistent improvement in classification accuracy throughout the training iterations, with low loss values for both the Textcat and Tok2Vec components. Therefore, considering the higher Cats Score and consistent performance improvement, the spaCy model can be deemed as performing better than the Prodigy model in this scenario.

There are several factors to consider when training and initialization to build a good model, our main area of focus was on the quality and cleanliness of the training data, as it plays a crucial role. Ensuring that the dataset is free from noise, duplicates, and inconsistencies allows the model to learn from accurate and reliable information, ultimately improving its performance. 
Additionally, the choice of features and representations used by the model, as well as the architecture and hyperparameters, significantly impact its ability to generalize well to unseen data. Overall, a systematic and comprehensive approach to training and initialization, combined with clean data and different feature choices, contributes to the development of the best-performing model.

## Project Proposal for a recipe classifier that can differentiate various cuisine types

The first step in developing this new recipe classifier would be to procure an appropriate dataset; an ideal dataset would contain a vast array of information pertaining to the preparation and composition of different foods throughout the world. These characteristics would ensure that our model is exposed to a variety of first-hand sources that could offer valuable advice or insights relating to recipes in their specific culture. Procuring a dataset with this wide of scope might be a bit difficult, so it is possible that the dataset will have to be constructed from multiple different sources; for example, it might be best to find a Reddit forum for each individual variety of cuisine that you want the model to be able to classify. This will ensure that the model has sufficient information on each variety of cuisine you are asking it to classify, as a lack of information on a specific cuisine could result in an inaccurate classifier. Once the dataset collection has been completed, we can start determining the categories/labels that the classifier will utilize; this step is highly dependent on the data collection process, as the breadth of the evaluation dataset chosen for the classifier will determine how many categories/types of cuisine the model can be applied to. This step is also important for ensuring that our model receives the necessary information to accurately classify recipes into distinct cuisine types, as every category must be represented somewhere in the evaluation dataset.

Once the dataset has been chosen and compiled, the next portion of the process will involve annotating the Evaluation Dataset by accurately labeling/categorizing each record based on what variety of cuisine it most closely relates to. This can be most easily completed using the “ngrok” package in a code editor like Visual Studio and a NER (Named Entity Recognition) software such as Prodigy. We will work to equip our model with a comprehensive understanding of the various cuisines we chose to include in our dataset, enabling it to make informed classifications with confidence. Additionally, if the annotation process was completed by multiple people, it might also involve combing through everyone's annotations and deciding which is most accurate; this will further improve the quality of your dataset and the accuracy of your model.

Finally, you can now move on to actually training the classifier; this part of the process will be a little different since we are now working with multiple classes. The main thing to prioritize is making sure that each class remains consistent and accurately captures any related records. This can be more easily accomplished by performing pre-processing steps such as tokenization to break sentences into pieces that are easier for the model to digest/understand. It might also include techniques such as language identification, as it is likely that cuisines belonging to a specific country/nationality will often be discussed in its native language. Experimenting with different techniques will provide further insight into how the model needs to be tweaked to maximize its accuracy and effectiveness, which can then be used to further improve the results of the classifier. It may be beneficial to utilize metrics such as Precision and Recall to determine where the model is particularly strong or weak, and then take action to improve the model in these specific areas.


### Link to our code in Google Colab
https://colab.research.google.com/drive/1kV-GH6NbNdJnuWHLmzrw-W--iXz28CK6?usp=sharing 
 


**I provided an example of what a labeled entry looked like to ChatGPT using the “labeled.jsonl” dataset and then asked it to construct and generate multi-class classifier that meets the requirements/description I provided in the 1-page project proposal and completes each of the steps I provided in the 1-page project proposal as well:**

```python
import jsonlines
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# Load dataset from .jsonl file
dataset = []
with jsonlines.open('your_dataset.jsonl') as reader:
    for record in reader:
        dataset.append(record)

# Convert dataset to pandas DataFrame
df = pd.DataFrame(dataset)

# Split dataset into features (X) and target labels (y)
X = df['text']
y = df['meta']['section']  # Assuming 'section' contains cuisine types

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construct a pipeline for text classification
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),  # Convert text to numerical features using TF-IDF
    ('clf', LinearSVC()),          # Use Linear Support Vector Classifier for classification
])

# Train the classifier on the training data
text_clf.fit(X_train, y_train)

# Predict labels for the test data
y_pred = text_clf.predict(X_test)

# Generate classification report
print(classification_report(y_test, y_pred))

```
