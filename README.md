# SpamSafe-Safeguarding-Your-SMS


**Title :**
        SMS Spam Classifier.


 **Project Summary:**
            Created a refined pipeline, starting with data sanitization and EDA for discerning patterns.
            Refines raw SMS data into informative vectors, discerning nuanced patterns to differentiate between    authentic and spam messages accurately.



## Technology used:
            NLP, KNN, ML,Label Encoder, Stacking, Bagging, XGBoost, AdaBoost, GradientBoost,
Decision Tree,Naive Bayes, GaussianNB, MNB, BNB, NLTK,SVM, TF-IDF etc.


The project utilizes various technologies to accomplish its tasks effectively:

1. **Python Programming Language**: Python serves as the primary language for coding the project due to its simplicity, extensive libraries, and compatibility with machine learning frameworks.

2. **Pandas and NumPy**: These libraries are used for data manipulation and numerical computations, enabling efficient data handling and processing.

3. **NLTK (Natural Language Toolkit)**: NLTK is employed for text preprocessing tasks such as tokenization, stemming, and stop word removal, facilitating the transformation of raw SMS data into structured format.

4. **Scikit-learn**: Scikit-learn provides a wide range of machine learning algorithms and tools for model building, training, and evaluation. It includes classifiers like Gaussian Naive Bayes, Support Vector Machines, and ensemble methods.

5. **Matplotlib and Seaborn**: These libraries are utilized for data visualization, enabling the exploration of data distributions, patterns, and relationships.

6. **WordCloud**: WordCloud is used to generate visual representations of the most frequent words in spam messages, aiding in understanding common themes and patterns.

7. **CountVectorizer and TfidfVectorizer**: These components from Scikit-learn are employed for converting text data into numerical feature vectors, crucial for training machine learning models.

8. **VotingClassifier and StackingClassifier**: These ensemble techniques from Scikit-learn are utilized to combine multiple base classifiers to improve predictive performance and robustness.

9. **XGBoost (Extreme Gradient Boosting)**: XGBoost is a powerful gradient boosting library used for building ensemble models, known for its speed and performance in handling large datasets.

10. **RandomForestClassifier**: This classifier from Scikit-learn is based on the random forest algorithm, which constructs multiple decision trees during training and outputs the mode of the classes as the prediction, providing robust classification performance.

By leveraging these technologies in tandem, the project achieves efficient data preprocessing, model training, and evaluation, ultimately resulting in accurate classification of spam and non-spam SMS messages.




## Project workflow:


The project operates through a series of coherent steps, each contributing to its overall functionality:

1. **Data Sanitization**:
   - Commencing with raw SMS data, the project initiates by cleaning the data, removing any irrelevant or redundant information. This process ensures that the subsequent analysis is performed on a refined dataset, free from noise or inconsistencies.

2. **Exploratory Data Analysis (EDA)**:
   - Following data sanitization, the project delves into exploratory data analysis. Through visualizations and statistical summaries, it seeks to uncover underlying patterns, distributions, and correlations within the SMS dataset. EDA provides valuable insights into the characteristics of spam and non-spam messages.

3. **Text Preprocessing**:
   - The raw SMS messages undergo extensive preprocessing, including tokenization, stemming, and stop word removal. This step transforms the text data into a standardized format, facilitating further analysis and model training.

4. **Feature Extraction**:
   - Utilizing techniques like CountVectorizer or TfidfVectorizer, the project converts the preprocessed text data into numerical feature vectors. These vectors represent the frequency or importance of words within each SMS message, serving as input for the machine learning models.

5. **Model Training**:
   - The refined feature vectors are then used to train a variety of machine learning models, ranging from Gaussian Naive Bayes to ensemble methods like Random Forest and XGBoost. Each model learns to distinguish between spam and non-spam messages based on the extracted features and labeled data.

6. **Model Evaluation**:
   - The trained models are evaluated using a separate test dataset to assess their performance in accurately classifying SMS messages. Metrics such as accuracy, precision, and recall are computed to gauge the efficacy of each model in identifying spam.

7. **Ensemble Techniques**:
   - To further enhance classification accuracy, ensemble techniques like Voting and Stacking are employed. These methods combine predictions from multiple base classifiers to produce a final consensual prediction, leveraging the strengths of individual models.

8. **Final Deployment**:
   - After thorough evaluation and fine-tuning, the most effective model or ensemble of models is selected for deployment. This final system is capable of automatically classifying incoming SMS messages as either spam or legitimate, providing users with a robust defense against unwanted communication.

By meticulously executing these steps, the project achieves its overarching goal of developing a reliable SMS spam classification system, effectively safeguarding users' communication channels.





 
**Conclusion:**
The conclusion summarizes the key findings of the research and emphasizes the efficacy of the proposed SMS spam classification system.
It underscores the significance of the project in mitigating the threat of SMS spam and enhancing communication security for users.

