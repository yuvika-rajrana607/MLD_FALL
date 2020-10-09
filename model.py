import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#importing the data
df_Train = pd.read_csv('training_pc.csv')

#removing missing values
df_Train = df_Train.dropna(axis=0)

#changing datatype
df_Train[['gleason_score', 'previous_cancer', 'smoker', 'rd_thrpy', 'h_thrpy', 'chm_thrpy', 'cry_thrpy', 'brch_thrpy',
          'rad_rem', 'race',
          't_score', 'n_score', 'm_score',
          'multi_thrpy', 'family_history', 'first_degree_history', 'stage', 'side','survival_7_years']] = df_Train[
    ['gleason_score', 'previous_cancer', 'smoker', 'rd_thrpy', 'h_thrpy', 'chm_thrpy', 'cry_thrpy', 'brch_thrpy',
     'rad_rem', 't_score',
     'n_score', 'm_score',
     'race', 'multi_thrpy', 'family_history', 'first_degree_history', 'stage', 'side','survival_7_years']].astype(
    object)

#removing unnecessary columns
df_Train_new = df_Train.drop(["tumor_diagnosis", "tumor_1_year", "psa_diagnosis", "psa_1_year", "first_degree_history","gleason_score"
                                 ,"t_score", "n_score", "m_score", "race", "first_degree_history", "symptoms", "h_thrpy",
                              "cry_thrpy", "multi_thrpy","brch_thrpy","height", "weight","side"],
                             axis=1)


#training and test data partition
x_prostat = df_Train_new[df_Train_new.columns[0:11]]
y_prostat = list(df_Train_new[df_Train_new.columns[11]])
X_train, X_test, y_train, y_test = train_test_split(x_prostat, y_prostat, test_size=.30)

#model building
rf = RandomForestClassifier(n_estimators=500, random_state=35, max_features=5, max_depth=5)
rf.fit(X_train, y_train).score(X_test, y_test)

#Saving model to disk
pickle.dump(rf, open('model.pkl','wb'))

#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[5, 80, 1, 1, 1, 15, 6, 2, 1, 1, 1]]))
#estimator.__getstate__()['_sklearn_version
