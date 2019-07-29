'''
This was exported straight from the attached Jupyter Notebook. Please refer 
to the notebook if you want to run the code more smoothly. 
'''

#!/usr/bin/env python
# coding: utf-8

# In[693]:


# NBA Hackathon Challenge
# Varun Ramakrishnan, Karman Cheema, Michael Abelar, Arjun Guru
# University of Pennsylvania


# In[694]:


# Just some imports
import re
import operator
import datefinder

import pandas as pd
import numpy as np

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# In[695]:


# Read in CSV as a dataframe
df = pd.read_csv("training_set.csv", encoding = 'unicode_escape')
dfCopy = df
# This is the dataframe that we'll be using for everything
dfForStats = dfCopy


# In[696]:


'''
Here's some preliminary work we did to understand the task at hand:
In short, we parse through the descriptions and calculate the Engagement / Follower ratio
for each tag (i.e. @kingjames) for each type of post. By doing so, we can confirm our hunch
that tags had a significant effect on engagement. In other words, more high profile players
and teams (LeBron, Steph Curry, Warriors, Cavs, Raptors) garner more engagement on each post
than other players/teams had. In addition, we found that albums and videos had far more 
engagement. These results can be seen in the output of the three sorted lists in the three 
cells below. 
'''
# Fill all the blank descriptions with "N/A" before starting
dfForStats = dfForStats.fillna("N/A")
frequency_photo = {}
engagements_photo = {}
frequency_video = {}
engagements_video = {}
frequency_album = {}
engagements_album = {}
# Parse through each row and calculate this ratio, looking at type and description
for i in range(7766):
    f = dfForStats.iloc[i]['Engagements'] / dfForStats.iloc[i]['Followers at Posting']
    descrip = re.findall(r'@\w*', dfForStats.iloc[i]['Description'])
    if dfForStats.iloc[i]['Type'] == 'Video':
        
        for name in descrip:
            if name in frequency_video:
                frequency_video[name] = frequency_video[name] + 1
                engagements_video[name] = engagements_video[name] + f
            else:
                frequency_video[name] = 1
                engagements_video[name] = f
        
    if dfForStats.iloc[i]['Type'] == 'Photo':
        for name in descrip:
            if name in frequency_photo:
                frequency_photo[name] = frequency_photo[name] + 1
                engagements_photo[name] = engagements_photo[name] + f
            else:
                frequency_photo[name] = 1
                engagements_photo[name] = f
    if dfForStats.iloc[i]['Type'] == 'Album':
        for name in descrip:
            if name in frequency_album:
                frequency_album[name] = frequency_album[name] + 1
                engagements_album[name] = engagements_album[name] + f
            else:
                frequency_album[name] = 1
                engagements_album[name] = f
sorted_album = sorted(engagements_album.items(), key=operator.itemgetter(1))
sorted_video = sorted(engagements_video.items(), key=operator.itemgetter(1))
sorted_photo = sorted(engagements_photo.items(), key=operator.itemgetter(1))


# In[697]:


sorted_album


# In[698]:


sorted_video


# In[699]:


sorted_photo


# In[700]:


# Find all tags in each description and number of tags per description
allTags = []
numTags = []
for index, row in dfForStats.iterrows():
    entry = row['Description']
    descrip = re.findall(r'@\w*', entry)
    if descrip:
        descripStr = ','.join(descrip)
        allTags.append(descripStr)
        numTags.append(len(descrip))
    else:
        allTags.append("")
        numTags.append(0)


# In[701]:


# Append both of these stats to the dataframe
dfForStats['Tags'] = allTags
dfForStats['# Tags'] = numTags


# In[702]:


# Find all hashtags in each description and number of hashtags per description
allHashtags = []
numHashtags = []
for index, row in dfForStats.iterrows():
    entry = row['Description']
    descrip = re.findall(r'#\w*', entry)
    if descrip:
        descripStr = ','.join(descrip)
        allHashtags.append(descripStr)
        numHashtags.append(len(descrip))
    else:
        allHashtags.append("")
        numHashtags.append(0)


# In[703]:


# Append both of these stats to the dataframe
dfForStats['Hashtags'] = allHashtags
dfForStats['# Hashtags'] = numHashtags


# In[704]:


'''
Date is also quite significant for engagement. Because most NBA games are played at night,
more engagement in found later in the day. In addition, April-June has far more engagement
as a result of the playoffs. Thus, we found it valuable to append all date values to the
dataframe. 
'''
months = []
days = []
hours = []
minutes = []
weekdays = []
for index, row in dfForStats.iterrows():
    entry = row['Created']
    matches = list(datefinder.find_dates(entry))
    theDate = matches[0]
    months.append(theDate.month)
    days.append(theDate.day)
    hours.append(theDate.hour)
    minutes.append(theDate.minute)
    weekdays.append(theDate.weekday())
# Append all of these stats to the dataframe
dfForStats['Month'] = months
dfForStats['Day'] = days
dfForStats['Hour'] = hours
dfForStats['Minute'] = minutes
dfForStats['WeekDay'] = weekdays


# In[705]:


# Split up the engagements
y = dfForStats["Engagements"]


# In[706]:


# Pseudo-One Hot Encoding for Tags, effectively creating a separate spot for each tag instead
# of one continuous string
for i in range(1,11):
    dfForStats["Tag" + str(i)] = ""
for i in range(7766):
    result = [x.strip() for x in allTags[i].split(',')]
    for j in range(len(result)):
        dfForStats.iloc[i, dfForStats.columns.get_loc("Tag" + str(j + 1))] = result[j]


# In[707]:


# Pseudo-One Hot Encoding for Hashtags ,effectively creating a separate spot for each hashtag 
# instead of one continuous string
for i in range(1,6):
    dfForStats["Hashtag" + str(i)] = ""
for i in range(7766):
    result = [x.strip() for x in allHashtags[i].split(',')]
    for j in range(len(result)):
        dfForStats.iloc[i, dfForStats.columns.get_loc("Hashtag" + str(j + 1))] = result[j]


# In[708]:


'''
This is a cool experiment we did - looking at the Instagram followers of each tag:
We thought that it may be useful to look at the number of followers of each tag in each
descriptions, with the notion that descriptions with the tags of users with lots of 
followers results in more engagement. 

Unfortunately, this did not improve the MAPE of our model and considering the immense amount
of time and memory this needed, it would not be prudent to include it. However, we thought 
it would be nice to include the code and the resulting output for reference. The output can 
be seen in the next cell
'''
'''
def followers(user):
    user = user
    url = 'https://www.instagram.com/'
    url_user = '%s%s%s' % (url, user, '/')
    url_login = 'https://www.instagram.com/accounts/login/ajax/'
    s = requests.Session()
    s.cookies.update ({'sessionid' : '', 'mid' : '', 'ig_pr' : '1',
                         'ig_vw' : '1920', 'csrftoken' : '',
                         's_network' : '', 'ds_user_id' : ''})
    login_post = {'username' : 'your_login',
                     'password' : 'your_pw'}
    s.headers.update ()
    r = s.get(url)
    s.headers.update({'X-CSRFToken' : r.cookies['csrftoken']})
    time.sleep(5 * random.random())
    login = s.post(url_login, data=login_post,
                      allow_redirects=True)
    s.headers.update({'X-CSRFToken' : login.cookies['csrftoken']})
    if login.status_code == 200:
        r = s.get('https://www.instagram.com/')
        finder = r.text.find('your_login')

    r = s.get(url_user)
    text = r.text

    finder_text_start = ('<script type="text/javascript">'
                         'window._sharedData = ')
    finder_text_start_len = len(finder_text_start)-1
    finder_text_end = ';</script>'

    all_data_start = text.find(finder_text_start)
    all_data_end = text.find(finder_text_end, all_data_start + 1)
    json_str = text[(all_data_start + finder_text_start_len + 1) \
                   : all_data_end]
    user_info = json.loads(json_str)
    follower_count = user_info['entry_data']['ProfilePage'][0]['graphql']['user']['edge_followed_by']['count']
    return follower_count


for i in range(1,11):
    dfForStats[str(i)] = ''
for i in range(0,7766):
    agg_followers = 0
    result = [x.strip() for x in allHashtags[i].split(',')]
    for j in range(len(result)):
        if result[0] != '':
            if result[j][1:] in cache_followers:
                agg_followers += cache_followers[result[j][1:]]
            else:
                try:
                    fol = followers(result[j][1:])
                    agg_followers += fol
                    cache_followers[result[j][1:]] = fol
                except:
                    pass
        dfForStats.iloc[i, dfForStats.columns.get_loc(str(j + 1))] = result[j]
    dfForStats.iloc[i, dfForStats.columns.get_loc('agg_followers')] = agg_followers
print("Done")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(dfForStats['agg_followers'])
'''


# In[709]:


# Drop engagements column and set it as y
dfForStats = dfForStats.drop("Engagements", axis=1)


# In[710]:


from sklearn.preprocessing import LabelEncoder
# We must use LabelEncoder in order to cache all the labels in the original training set
# By doing so, we can use OneHotEncoder and be prepared for never-before-seen labels
# in the holdout set
feat_dict = {}
for col in dfForStats.columns:
    feat_dict[col] = LabelEncoder().fit(dfForStats[col])
    dfForStats[col] = feat_dict[col].transform(dfForStats[col])


# In[711]:


# Using One Hot Encoder for more features
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(dfForStats)
dfForStats = enc.transform(dfForStats).toarray()


# In[712]:


dfForStats


# In[713]:


# Split into testing and training
X_train, X_test, y_train, y_test = train_test_split(dfForStats, y, test_size=0.20, random_state=42)


# In[714]:


'''
MODEL SELECTION: Linear Regression
We tried many models, including Random Forest and multiple neural networks (using Keras), but found that
a simple, vanilla Linear Regression resulted in the lowest MAPE. Feature engineering was much more 
important in our case. 
'''
# Linear Regression
lin_reg = LinearRegression().fit(X_train, y_train)
predictions = lin_reg.predict(X_test)
error = mean_absolute_error(y_test, predictions)
error


# In[715]:


# Helper function to calculate MAPE
def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


# In[716]:


error = smape(y_test, predictions)
print(error)
# Final mape of 5.847%


# In[717]:


# Take the hold out data and get predictions
testDF = pd.read_csv("holdout_set.csv", encoding = 'unicode_escape')
# Create copy for exporting csv
testDFCopy = testDF.copy()


# In[718]:


'''
In all of the cells below, we basically recreate the feature engineering performed on the training set 
in order to obtain the correct input shape for obtaining the results for the holdout set
'''


# In[719]:


# Find all tags in each description
allTags = []
numTags = []
maxNum = 0
for index, row in testDF.iterrows():
    entry = row['Description']
    descrip = re.findall(r'@\w*', entry)
    if descrip:
        descripStr = ','.join(descrip)
        allTags.append(descripStr)
        numTags.append(len(descrip))
        if (len(descrip) > maxNum):
            maxNum = len(descrip)
    else:
        allTags.append("")
        numTags.append(0)


# In[720]:


# Add fields to dataframe
testDF['Tags'] = allTags
testDF['# Tags'] = numTags


# In[721]:


# Find all hashtags in each description
allHashtags = []
numHashtags = []
maxNum = 0
for index, row in testDF.iterrows():
    entry = row['Description']
    descrip = re.findall(r'#\w*', entry)
    if descrip:
        descripStr = ','.join(descrip)
        allHashtags.append(descripStr)
        numHashtags.append(len(descrip))
        if (len(descrip) > maxNum):
            maxNum = len(descrip)
    else:
        allHashtags.append("")
        numHashtags.append(0)


# In[722]:


# Add fields to dataframe
testDF['Hashtags'] = allHashtags
testDF['# Hashtags'] = numHashtags


# In[723]:


# Drop engagements because it's all empty!
testDF = testDF.drop(["Engagements"], axis=1)


# In[724]:


# Extract important date features
months = []
days = []
hours = []
minutes = []
weekdays = []
for index, row in testDF.iterrows():
    entry = row['Created']
    matches = list(datefinder.find_dates(entry))
    theDate = matches[0]
    months.append(theDate.month)
    days.append(theDate.day)
    hours.append(theDate.hour)
    minutes.append(theDate.minute)
    weekdays.append(theDate.weekday())
    
# Add fields to dataframe
testDF['Month'] = months
testDF['Day'] = days
testDF['Hour'] = hours
testDF['Minute'] = minutes
testDF['WeekDay'] = weekdays


# In[725]:


# Pseudo-One Hot Encoding for Tags
for i in range(1,11):
    testDF["Tag" + str(i)] = ""
for i in range(1000):
    result = [x.strip() for x in allTags[i].split(',')]
    for j in range(len(result)):
        testDF.iloc[i, testDF.columns.get_loc("Tag" + str(j + 1))] = result[j]


# In[726]:


# Pseudo-One Hot Encoding for Hashtags
for i in range(1,6):
    testDF["Hashtag" + str(i)] = ""
for i in range(1000):
    result = [x.strip() for x in allHashtags[i].split(',')]
    for j in range(len(result)):
        testDF.iloc[i, testDF.columns.get_loc("Hashtag" + str(j + 1))] = result[j]


# In[727]:


# This enables us to make sure the width of the input data is the same for both testing and training
for col in testDF.columns:
    testDF[col] = feat_dict[col].fit_transform(testDF[col])

testDF = enc.transform(testDF).toarray()


# In[728]:


# Train final model that contains all of the training data
lin_reg_final = LinearRegression().fit(dfForStats, y)


# In[729]:


# Obtain predictions
holdout_predictions = lin_reg_final.predict(testDF)


# In[730]:


# Round each digit in prediction list to nearest integer
# Since engagements must be a whole number!
for i in range(len(holdout_predictions)):
    holdout_predictions[i] = int(round(holdout_predictions[i]))


# In[731]:


# Prepare for output of data
outputTestDF = testDFCopy.copy()
outputTestDF["Engagements"] = holdout_predictions


# In[732]:


# Save final submission CSV
holdout_set_submission = outputTestDF.to_csv(r'holdout_set_submission.csv', index = None, header=True) 


# In[ ]:




