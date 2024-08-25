

# # Multinomial Classification (normal or DOS or PROBE or R2L or U2R)




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


import itertools
import seaborn as sns
import pandas_profiling
import statsmodels.formula.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices

from pydantic.v1 import BaseSettings

# In[3]:


from sklearn import datasets
from sklearn.feature_selection import RFE
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif


# In[4]:


train=pd.read_csv('NSL_Dataset\Train.txt',sep=',')
test=pd.read_csv('NSL_Dataset\Test.txt',sep=',')


# In[5]:


train.head()


# In[6]:


columns=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
         "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations", 
         "num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate",
         "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
         "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
         "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
         "dst_host_srv_rerror_rate","attack","last_flag"] 


# In[7]:


len(columns)


# In[8]:


train.columns=columns
test.columns=columns


# In[9]:


train.head()


# In[10]:


test.head()


# In[11]:


train.info()


# In[12]:


test.info()


# In[13]:


train.describe().T


# In attack_class normal means 0, DOS means 1, PROBE means 2, R2L means 3 and U2R means 4.

# In[14]:


train.loc[train.attack=='normal','attack_class']=0

train.loc[(train.attack=='back') | (train.attack=='land') | (train.attack=='pod') | (train.attack=='neptune') | 
         (train.attack=='smurf') | (train.attack=='teardrop') | (train.attack=='apache2') | (train.attack=='udpstorm') | 
         (train.attack=='processtable') | (train.attack=='worm') | (train.attack=='mailbomb'),'attack_class']=1

train.loc[(train.attack=='satan') | (train.attack=='ipsweep') | (train.attack=='nmap') | (train.attack=='portsweep') | 
          (train.attack=='mscan') | (train.attack=='saint'),'attack_class']=2

train.loc[(train.attack=='guess_passwd') | (train.attack=='ftp_write') | (train.attack=='imap') | (train.attack=='phf') | 
          (train.attack=='multihop') | (train.attack=='warezmaster') | (train.attack=='warezclient') | (train.attack=='spy') | 
          (train.attack=='xlock') | (train.attack=='xsnoop') | (train.attack=='snmpguess') | (train.attack=='snmpgetattack') | 
          (train.attack=='httptunnel') | (train.attack=='sendmail') | (train.attack=='named'),'attack_class']=3

train.loc[(train.attack=='buffer_overflow') | (train.attack=='loadmodule') | (train.attack=='rootkit') | (train.attack=='perl') | 
          (train.attack=='sqlattack') | (train.attack=='xterm') | (train.attack=='ps'),'attack_class']=4


# In[15]:


test.loc[test.attack=='normal','attack_class']=0

test.loc[(test.attack=='back') | (test.attack=='land') | (test.attack=='pod') | (test.attack=='neptune') | 
         (test.attack=='smurf') | (test.attack=='teardrop') | (test.attack=='apache2') | (test.attack=='udpstorm') | 
         (test.attack=='processtable') | (test.attack=='worm') | (test.attack=='mailbomb'),'attack_class']=1

test.loc[(test.attack=='satan') | (test.attack=='ipsweep') | (test.attack=='nmap') | (test.attack=='portsweep') | 
          (test.attack=='mscan') | (test.attack=='saint'),'attack_class']=2

test.loc[(test.attack=='guess_passwd') | (test.attack=='ftp_write') | (test.attack=='imap') | (test.attack=='phf') | 
          (test.attack=='multihop') | (test.attack=='warezmaster') | (test.attack=='warezclient') | (test.attack=='spy') | 
          (test.attack=='xlock') | (test.attack=='xsnoop') | (test.attack=='snmpguess') | (test.attack=='snmpgetattack') | 
          (test.attack=='httptunnel') | (test.attack=='sendmail') | (test.attack=='named'),'attack_class']=3

test.loc[(test.attack=='buffer_overflow') | (test.attack=='loadmodule') | (test.attack=='rootkit') | (test.attack=='perl') | 
          (test.attack=='sqlattack') | (test.attack=='xterm') | (test.attack=='ps'),'attack_class']=4


# In[16]:


train.head()


# In[17]:


train.shape


# In[20]:


output=pandas_profiling.ProfileReport(train)
output


# ### Exporting pandas profiling output to html file

# In[22]:


output.to_file('pandas_profiling.html')


# ### Basic Exploratory Analysis

# In[23]:


# Protocol type distribution
plt.figure(figsize=(6,3))
sns.countplot(x="protocol_type", data=train)
plt.show()


# In[24]:


# service distribution
plt.figure(figsize=(6,10))
sns.countplot(y="service", data=train)
plt.show()


# In[25]:


# flag distribution
plt.figure(figsize=(6,3))
sns.countplot(x="flag", data=train)
plt.show()


# In[26]:


# attack distribution
plt.figure(figsize=(6,4))
sns.countplot(y="attack", data=train)
plt.show()


# In[27]:


# attack class distribution
plt.figure(figsize=(6,3))
sns.countplot(x="attack_class", data=train)
plt.show()


# #### identifying relationships (between Y & numerical independent variables by comparing means)

# In[28]:


train.groupby('attack_class').mean().T


# ##### Observations:
# - The length of time duration of connection for attack is higher than  normal.
# - Wrong fragments in the connection is only present in attack.
# - Number of outbound commands in an ftp session  are 0 in both normal and attack.

# In[29]:


numeric_var_names=[key for key in dict(train.dtypes) if dict(train.dtypes)[key] in ['float64', 'int64', 'float32', 'int32']]
cat_var_names=[key for key in dict(train.dtypes) if dict(train.dtypes)[key] in ['object', 'O']]


# In[30]:


numeric_var_names


# In[31]:


cat_var_names


# In[32]:


train_num=train[numeric_var_names]
test_num=test[numeric_var_names]
train_num.head(5)


# In[33]:


train_cat=train[cat_var_names]
test_cat=test[cat_var_names]
train_cat.head(5)


# ### Data Audit Report

# In[34]:


# Creating Data audit Report
def var_summary(x):
    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(), x.dropna().quantile(0.01), x.dropna().quantile(0.05),x.dropna().quantile(0.10),x.dropna().quantile(0.25),x.dropna().quantile(0.50),x.dropna().quantile(0.75), x.dropna().quantile(0.90),x.dropna().quantile(0.95), x.dropna().quantile(0.99),x.max()], 
                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])

num_summary=train_num.apply(lambda x: var_summary(x)).T


# In[35]:


num_summary


# In[36]:


num_summary.to_csv('num_summary.csv')


# ### Handling Outlier

# In[37]:


#Handling Outliers
def outlier_capping(x):
    x = x.clip(upper=x.quantile(0.99))
    x = x.clip(lower=x.quantile(0.01))
    return x

train_num=train_num.apply(outlier_capping)


# #### No missing in train dataset . So , Missing treatment not required .

# In[38]:


def cat_summary(x):
    return pd.Series([x.count(), x.isnull().sum(), x.value_counts()], 
                  index=['N', 'NMISS', 'ColumnsNames'])

cat_summary=train_cat.apply(cat_summary)


# In[39]:


cat_summary


# ### Dummy Variable Creation

# In[40]:


# An utility function to create dummy variable
def create_dummies( df, colname ):
    col_dummies = pd.get_dummies(df[colname], prefix=colname, drop_first=True)
    df = pd.concat([df, col_dummies], axis=1)
    df.drop( colname, axis = 1, inplace = True )
    return(df)


# In[41]:


#for c_feature in categorical_features
for c_feature in ['protocol_type', 'service', 'flag', 'attack']:
    train_cat = create_dummies(train_cat,c_feature)
    test_cat = create_dummies(test_cat,c_feature)
train_cat.head()


# ### Final file for analysis

# In[42]:


train_new = pd.concat([train_num, train_cat], axis=1)
test_new = pd.concat([test_num, test_cat], axis=1)
train_new.head()


# In[43]:


# correlation matrix (ranges from 1 to -1)
corrm=train_new.corr()
corrm


# In[44]:


corrm.to_csv('corrm.csv')


# In[45]:


# visualize correlation matrix in Seaborn using a heatmap
sns.heatmap(corrm)


# #### Dropping columns based on data audit report
#         - Based on low variance (near zero variance)
#         - High missings (>25% missings)
#         - High correlations between two numerical variables

# In[46]:


train_new.drop(columns=['land','wrong_fragment','urgent','num_failed_logins',"root_shell","su_attempted","num_root",
                        "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
                        'dst_host_rerror_rate','dst_host_serror_rate','dst_host_srv_rerror_rate','dst_host_srv_serror_rate',
                        'num_root','num_outbound_cmds','srv_rerror_rate','srv_serror_rate'], axis=1, inplace=True)


# In[47]:


sns.heatmap(train_new.corr())


# #### Variable reduction using Select K-Best technique

# In[48]:


X = train_new[train_new.columns.difference(['attack_class'])]
X_new = SelectKBest(f_classif, k=15).fit(X, train_new['attack_class'] )


# In[49]:


X_new.get_support()


# In[50]:


X_new.scores_


# In[51]:


# capturing the important variables
KBest_features=X.columns[X_new.get_support()]
KBest_features


# ### Final list of variable selected for the model building using Select KBest

# attack_neptune, attack_normal, attack_satan, count, dst_host_diff_srv_rate, dst_host_same_src_port_rate, dst_host_same_srv_rate, dst_host_srv_count, flag_S0, flag_SF, last_flag, logged_in, same_srv_rate, serror_rate, service_http

# In[52]:


train=train_new
test=test_new


# ## Model Building

# In[53]:


top_features=['attack_neptune','attack_normal','attack_satan','count','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_same_srv_rate','dst_host_srv_count','flag_S0','flag_SF','last_flag','logged_in','same_srv_rate','serror_rate','service_http']
X_train = train[top_features]
y_train = train['attack_class']
X_test = test[top_features]
y_test = test['attack_class']


# ### Building logistic Regression

# #### 1) LogisticRegression

# In[54]:


lr_clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, y_train)


# In[56]:


y_pred=lr_clf.predict(X_test)
y_pred


# In[57]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# #### 2) RidgeClassifier

# In[58]:


from sklearn.linear_model import RidgeClassifier


# In[59]:


rc_clf = RidgeClassifier().fit(X_train, y_train)


# In[60]:


y_pred=rc_clf.predict(X_test)
y_pred


# In[61]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# ### K-Nearest Neighbors

# #### 1) KNeighborsClassifier

# In[62]:


from sklearn.neighbors import KNeighborsClassifier


# In[63]:


k_neigh = KNeighborsClassifier(n_neighbors=3)
k_neigh.fit(X_train, y_train)


# In[64]:


y_pred=k_neigh.predict(X_test)
y_pred


# In[65]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# #### 3) NearestCentroid

# In[75]:


from sklearn.neighbors.nearest_centroid import NearestCentroid


# In[76]:


nc = NearestCentroid()
nc.fit(X_train, y_train)


# In[77]:


y_pred=nc.predict(X_test)
y_pred


# In[78]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# ### Discriminant Analysis

# #### 1) LinearDiscriminantAnalysis

# In[79]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[80]:


lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train) 


# In[81]:


y_pred=lda.predict(X_test)
y_pred


# In[82]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# #### 2) QuadraticDiscriminantAnalysis

# In[83]:


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# In[84]:


qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)


# In[85]:


y_pred=qda.predict(X_test)
y_pred


# In[86]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# ### Decision Trees

# In[87]:


from sklearn.model_selection import cross_val_score
from sklearn import metrics
import sklearn.tree as dt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz, export
from sklearn.model_selection import GridSearchCV


# In[88]:


clf_tree = DecisionTreeClassifier( max_depth = 5)
clf_tree=clf_tree.fit( X_train, y_train )


# In[89]:


y_pred=qda.predict(X_test)
y_pred


# In[90]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# #### Fine Tuning the parameters

# In[91]:


param_grid = {'max_depth': np.arange(3, 9),
             'max_features': np.arange(3,9)}


# In[92]:


tree = GridSearchCV(DecisionTreeClassifier(), param_grid, cv = 5)
tree.fit( X_train, y_train )


# In[93]:


tree.best_score_


# In[94]:


tree.best_estimator_


# In[95]:


tree.best_params_


# ### Building Final Decision Tree Model

# In[96]:


clf_tree = DecisionTreeClassifier( max_depth = 8, max_features=8 )
clf_tree.fit( X_train, y_train )


# #### Feature Relative Importance

# In[97]:


clf_tree.feature_importances_


# In[98]:


# summarize the selection of the attributes
import itertools
feature_map = [(i, v) for i, v in itertools.zip_longest(X_train.columns, clf_tree.feature_importances_)]

feature_map


# In[99]:


Feature_importance = pd.DataFrame(feature_map, columns=['Feature', 'importance'])
Feature_importance.sort_values('importance', inplace=True, ascending=False)
Feature_importance


# In[100]:


tree_test_pred = pd.DataFrame( { 'actual':  y_test,
                            'predicted': clf_tree.predict( X_test ) } )


# In[101]:


tree_test_pred.sample( n = 10 )


# In[104]:


accuracy_score( tree_test_pred.actual, tree_test_pred.predicted )


# In[103]:


tree_cm = metrics.confusion_matrix( tree_test_pred.predicted,
                                 tree_test_pred.actual,
                                 [1,0] )
sns.heatmap(tree_cm, annot=True,
         fmt='.2f',
         xticklabels = ["Left", "No Left"] , yticklabels = ["Left", "No Left"] )

plt.ylabel('True label')
plt.xlabel('Predicted label')


# ## Naive Bayes Model

# #### 1) BernoulliNB

# In[105]:


from sklearn.naive_bayes import BernoulliNB


# In[106]:


bnb_clf = BernoulliNB()
bnb_clf.fit(X_train, y_train)


# In[107]:


y_pred=bnb_clf.predict(X_test)
y_pred


# In[108]:


nb_cm = metrics.confusion_matrix( y_test,y_pred )
sns.heatmap(nb_cm, annot=True,  fmt='.2f', xticklabels = ["no", "Yes"] , yticklabels = ["No", "Yes"] )
plt.ylabel('True label')
plt.xlabel('Predicted label')


# In[109]:


accuracy_score( y_test, y_pred )


# #### 2) GaussianNB

# In[110]:


from sklearn.naive_bayes import GaussianNB


# In[111]:


gnb_clf = GaussianNB()
gnb_clf.fit(X_train, y_train)


# In[112]:


y_pred=gnb_clf.predict(X_test)
y_pred


# In[113]:


nb_cm = metrics.confusion_matrix( y_test, y_pred )
sns.heatmap(nb_cm, annot=True,  fmt='.2f', xticklabels = ["no", "Yes"] , yticklabels = ["No", "Yes"] )
plt.ylabel('True label')
plt.xlabel('Predicted label')


# In[114]:


accuracy_score( y_test, y_pred )


# ### Support Vector Machine (SVM)

# #### 1) LinearSVC

# In[115]:


from sklearn.svm import LinearSVC


# In[116]:


svm_clf = LinearSVC(random_state=0, tol=1e-5)
svm_clf.fit(X_train, y_train)


# In[117]:


y_pred=svm_clf.predict(X_test)
y_pred


# In[118]:


accuracy_score( y_test, y_pred )


# #### 2) SVC

# In[119]:


from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

model = SVC(kernel='rbf', class_weight='balanced',gamma='scale')


# In[120]:


model.fit(X_train, y_train)


# In[121]:


y_pred=model.predict(X_test)
y_pred


# In[122]:


accuracy_score( y_test, y_pred )


# ### Stochastic Gradient Descent (SGD)

# In[123]:


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# In[124]:


model = SGDClassifier(loss="hinge", penalty="l2")
model.fit(X_train, y_train)


# In[125]:


y_pred=model.predict(X_test)
y_pred


# In[126]:


accuracy_score( y_test, y_pred )


# In[127]:


n_iters = [5, 10, 20, 50, 100, 1000]
scores = []
for n_iter in n_iters:
    model = SGDClassifier(loss="hinge", penalty="l2", max_iter=n_iter)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))

plt.title("Effect of n_iter")
plt.xlabel("n_iter")
plt.ylabel("score")
plt.plot(n_iters, scores) 


# In[128]:


# losses
losses = ["hinge", "log", "modified_huber", "perceptron", "squared_hinge"]
scores = []
for loss in losses:
    model = SGDClassifier(loss=loss, penalty="l2", max_iter=1000)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))

plt.xlabel("loss")
plt.ylabel("score")
plt.title("Effect of loss")
x = np.arange(len(losses))
plt.xticks(x, losses)
plt.plot(x, scores)


# In[129]:


from sklearn.model_selection import GridSearchCV

params = {
    "loss" : ["hinge", "log", "squared_hinge", "modified_huber"],
    "alpha" : [0.0001, 0.001, 0.01, 0.1],
    "penalty" : ["l2", "l1", "none"],
}

model = SGDClassifier(max_iter=100)
clf = GridSearchCV(model, param_grid=params)


# In[130]:


clf.fit(X_train, y_train)
print(clf.best_score_)


# In[131]:


y_pred=clf.predict(X_test)
y_pred


# In[132]:


accuracy_score( y_test, y_pred )


# ### Neural Network Model

# In[133]:


from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)


# In[134]:


# Now apply the transformations to the data:
train_X = scaler.transform(X_train)
test_X = scaler.transform(X_test)


# In[135]:


mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
mlp.fit(train_X,y_train)


# In[136]:


y_pred=mlp.predict(test_X)
y_pred


# In[137]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,y_pred))


# In[138]:


print(classification_report(y_test,y_pred))


# In[139]:


mlp.coefs_


# In[140]:


accuracy_score( y_test, y_pred )


# ## Combine Model Predictions Into Ensemble Predictions
# 
# The three most popular methods for combining the predictions from different models are:
# 
# Bagging-> Building multiple models (typically of the same type) from different subsamples of the training dataset.
# 
# Boosting-> Building multiple models (typically of the same type) each of which learns to fix the prediction errors of a prior model in the chain.
# 
# Voting-> Building multiple models (typically of differing types) and simple statistics (like calculating the mean) are used to combine predictions.

# ### Bagging Algorithms
# 
# Bootstrap Aggregation or bagging involves taking multiple samples from your training dataset (with replacement) and training a model for each sample.
# 
# The final output prediction is averaged across the predictions of all of the sub-models.
# 
# The three bagging models covered in this section are as follows:
# 
# 1) Bagged Decision Trees
# 
# 2) Random Forest
# 
# 3) Extra Trees

# #### 1. Bagged Decision Trees
# Bagging performs best with algorithms that have high variance. A popular example are decision trees, often constructed without pruning.

# In[141]:


from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


# In[142]:


seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
print(results.mean())


# In[143]:


model.fit(X_train, y_train)


# In[144]:


y_pred=model.predict(X_test)
y_pred


# In[145]:


accuracy_score( y_test, y_pred )


# #### 2. Random Forest
# Random forest is an extension of bagged decision trees.

# In[146]:


from sklearn.ensemble import RandomForestClassifier


# In[147]:


seed = 7
num_trees = 100
max_features = 3
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
print(results.mean())


# In[148]:


model.fit(X_train, y_train)


# In[149]:


y_pred=model.predict(X_test)
y_pred


# In[150]:


accuracy_score( y_test, y_pred )


# #### 3. Extra Trees
# Extra Trees are another modification of bagging where random trees are constructed from samples of the training dataset.

# In[151]:


from sklearn.ensemble import ExtraTreesClassifier


# In[152]:


seed = 7
num_trees = 100
max_features = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
print(results.mean())


# In[153]:


model.fit(X_train, y_train)


# In[154]:


y_pred=model.predict(X_test)
y_pred


# In[155]:


accuracy_score( y_test, y_pred )


# ### Boosting Algorithms
# 
# Boosting ensemble algorithms creates a sequence of models that attempt to correct the mistakes of the models before them in the sequence.
# 
# Once created, the models make predictions which may be weighted by their demonstrated accuracy and the results are combined to create a final output prediction.
# 
# The two most common boosting ensemble machine learning algorithms are:
# 
# 1) AdaBoost
# 
# 2) Stochastic Gradient Boosting

# #### 1. AdaBoost
# 
# AdaBoost was perhaps the first successful boosting ensemble algorithm. It generally works by weighting instances in the dataset by how easy or difficult they are to classify, allowing the algorithm to pay or or less attention to them in the construction of subsequent models.

# In[156]:


from sklearn.ensemble import AdaBoostClassifier


# In[157]:


seed = 7
num_trees = 30
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
print(results.mean())


# In[158]:


model.fit(X_train, y_train)


# In[159]:


y_pred=model.predict(X_test)
y_pred


# In[160]:


accuracy_score( y_test, y_pred )


# #### 2. Stochastic Gradient Boosting
# Stochastic Gradient Boosting (also called Gradient Boosting Machines) are one of the most sophisticated ensemble techniques. It is also a technique that is proving to be perhaps of the the best techniques available for improving performance via ensembles.

# In[161]:


from sklearn.ensemble import GradientBoostingClassifier


# In[162]:


seed = 7
num_trees = 100
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
print(results.mean())


# In[163]:


model.fit(X_train, y_train)


# In[164]:


y_pred=model.predict(X_test)
y_pred


# In[165]:


accuracy_score( y_test, y_pred )


# ### Voting Ensemble
# 
# Voting is one of the simplest ways of combining the predictions from multiple machine learning algorithms.
# 
# It works by first creating two or more standalone models from your training dataset. A Voting Classifier can then be used to wrap your models and average the predictions of the sub-models when asked to make predictions for new data.
# 
# The predictions of the sub-models can be weighted, but specifying the weights for classifiers manually or even heuristically is difficult. More advanced methods can learn how to best weight the predictions from submodels, but this is called stacking (stacked generalization) and is currently not provided in scikit-learn.

# In[166]:


from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier


# In[167]:


seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X_train, y_train, cv=kfold)
print(results.mean())


# In[168]:


ensemble.fit(X_train, y_train)


# In[169]:


y_pred=ensemble.predict(X_test)
y_pred


# In[170]:


accuracy_score( y_test, y_pred )


# # Save Model

# In[171]:


import pickle
# Saving model to disk of random forest
pickle.dump(lr_clf, open('model.pkl','wb'))


# # Load Model and Predict

# In[172]:


import pickle
model=pickle.load(open('model.pkl', 'rb'))
model.predict([[1,0,0,229,0.06,0.00,0.04,10,0,0,21,0,0.04,0.00,0]])


# In[ ]:




