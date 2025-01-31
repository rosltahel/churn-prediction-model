
<h1>Churn Prediction Model</h1>

<p>This project is designed to predict customer churn for a telecom company using machine learning techniques. The goal is to help businesses identify customers who are likely to churn, allowing them to take proactive actions to retain them.</p>

<h2>Table of Contents</h2>
<ul>
  <li><a href="#introduction">Introduction</a></li>
  <li><a href="#dataset">Dataset</a></li>
  <li><a href="#installation">Installation</a></li>
  <li><a href="#usage">Usage</a></li>
  <li><a href="#model-evaluation">Model Evaluation</a></li>
  <li><a href="#technologies-used">Technologies Used</a></li>
  <li><a href="#license">License</a></li>
</ul>

<h2 id="introduction">Introduction</h2>

<p>Customer churn prediction is a crucial task for businesses, especially in the telecom industry. By predicting churn, companies can target high-risk customers and provide retention strategies. This project leverages machine learning algorithms to predict churn and evaluate model performance using several metrics.</p>

<h2 id="dataset">Dataset</h2>

<p>The dataset used in this project is the <a href="https://www.bigml.com">Churn Dataset from BigML</a>. It contains information about customers such as account length, service plans, usage, and whether the customer churned or not.</p>

<h3>Key Features:</h3>
<ul>
  <li><strong>State</strong>: The state in which the customer resides</li>
  <li><strong>International plan</strong>: Whether the customer has an international calling plan</li>
  <li><strong>Voice mail plan</strong>: Whether the customer subscribes to a voicemail plan</li>
  <li><strong>Churn</strong>: Whether the customer has churned (1) or not (0)</li>
  <li><strong>Total day minutes, Total eve minutes, Total night minutes</strong>: Duration of calls made during different times of the day</li>
  <li><strong>Customer service calls</strong>: Number of calls made to customer service</li>
</ul>

<h2 id="installation">Installation</h2>

<p>To run this project locally, clone the repository and install the required dependencies.</p>

<pre>
1. Clone the repository:
   git clone https://github.com/rosltahel/churn-prediction-model.git
   cd churn-prediction-model

2. Install the dependencies using pip:
   pip install -r requirements.txt
</pre>

<h2 id="usage">Usage</h2>

<h3>1. Load and Explore the Data</h3>
<pre>
import pandas as pd

df = pd.read_csv('churn-bigml-20.csv')
print(df.head())
</pre>

<h3>2. Preprocess the Data</h3>
<ul>
  <li>Encode categorical features using LabelEncoder</li>
  <li>Handle missing values if any</li>
  <li>Split the data into features (X) and target (y)</li>
</ul>

<h3>3. Model Training</h3>

<p>We use various machine learning models, including Logistic Regression and Random Forest, to predict churn.</p>

<pre>
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
</pre>

<h3>4. Evaluate the Model</h3>

<p>We evaluate the model performance using metrics like accuracy, confusion matrix, and classification report.</p>

<pre>
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
</pre>

<h2 id="model-evaluation">Model Evaluation</h2>

<p>The performance of the model was evaluated on the test set using several metrics:</p>
<ul>
  <li><strong>Accuracy</strong>: The overall accuracy of the model in predicting churn.</li>
  <li><strong>Confusion Matrix</strong>: A matrix showing the true positives, false positives, true negatives, and false negatives.</li>
  <li><strong>Precision, Recall, F1-Score</strong>: These metrics provide deeper insights into model performance, especially for the imbalanced classes.</li>
</ul>

<h3>After SMOTE & Hyperparameter Tuning:</h3>
<ul>
  <li><strong>Accuracy</strong>: 92%</li>
  <li><strong>Precision for Churned Customers</strong>: 0.67</li>
  <li><strong>Recall for Churned Customers</strong>: 0.53</li>
  <li><strong>F1-Score for Churned Customers</strong>: 0.59</li>
</ul>

<h2 id="technologies-used">Technologies Used</h2>

<ul>
  <li><strong>Python</strong>: Programming language used for data analysis and model building.</li>
  <li><strong>pandas</strong>: Data manipulation and analysis.</li>
  <li><strong>scikit-learn</strong>: Machine learning algorithms for training, evaluation, and model optimization.</li>
  <li><strong>SMOTE</strong>: Synthetic Minority Over-sampling Technique for handling class imbalance.</li>
  <li><strong>matplotlib & seaborn</strong>: Libraries for visualizations.</li>
</ul>

<h2 id="license">License</h2>

<p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>
