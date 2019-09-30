
# coding: utf-8

# # Recommendations with IBM
# 
# The project analyses the user interactions with teh articles on the IBM Watson Studio platform and make recommendations  about new articles based on the outcome of our analyses.
# 
# 
# We shall  build  different methods for making recommendations that can be used for different situations. 
# 
# **I. Rank Based Recommendations**
# 
# To get started in building recommendations, we will first find the most popular articles simply based on the most interactions. Since there are no ratings for any of the articles, it is easy to assume the articles with the most interactions are the most popular. These are then the articles we might recommend to new users (or anyone depending on what we know about them).
# 
# **II. User-User Based Collaborative Filtering**
# In order to build better recommendations for the users of IBM's platform, we could look at users that are similar in terms of the items they have interacted with. These items could then be recommended to the similar users. This would be a step in the right direction towards more personal recommendations for the users. 
# 
# **III. Content Based Recommendations (Future scope)**
# Given the amount of content available for each article, there are a number of different ways in which someone might choose to implement a content based recommendations system. Using NLP skills, we can come up with some extremely creative ways to develop a content based recommendation system. 
# 
# **IV. Matrix Factorization**
# Finally, complete a machine learning approach to building recommendations. Using the user-item interactions, we will build a matrix decomposition. Using decomposition, you will get an idea of how well we can predict new articles an individual might interact with (spoiler alert - it isn't great). We will finally discuss which methods can be used moving forward, and how to test how well the recommendations are working for engaging users.
# 
# 
# 
# ## Table of Contents
# 
# I. [Exploratory Data Analysis](#Exploratory-Data-Analysis)<br>
# II. [Rank Based Recommendations](#Rank)<br>
# III. [User-User Based Collaborative Filtering](#User-User)<br>
# IV. [Content Based Recommendations (EXTRA - NOT REQUIRED)](#Content-Recs)<br>
# V. [Matrix Factorization](#Matrix-Fact)<br>
# VI. [Extras & Concluding](#conclusions)
# 

# **Import libraries and data files**

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import project_tests as t
import pickle

get_ipython().magic(u'matplotlib inline')

df = pd.read_csv('data/user-item-interactions.csv')
df_content = pd.read_csv('data/articles_community.csv')
del df['Unnamed: 0']
del df_content['Unnamed: 0']

# Show df to get an idea of the data
df.head()


# In[2]:

# Show df_content to get an idea of the data
df_content.head()


# ### <a class="anchor" id="Exploratory-Data-Analysis">Part I : Exploratory Data Analysis</a>
# 
# Insight into the descriptive statistics of the data.
# 
# `1.` What is the distribution of how many articles a user interacts with in the dataset?  Provide a visual and descriptive statistics to assist with giving a look at the number of times each user interacts with an article.  

# In[3]:

#unique items/articles
len(df.article_id.unique())


# In[4]:

df.shape ,df_content.shape


# In[5]:

#unique users
len(df.email.unique())


# In[6]:

# missing users
df.email.isnull().sum()


# In[7]:

#missing users df
df[df.email.isnull()].head()


# In[8]:

#Distribution of users
usr_item = df.groupby('email')['article_id'].count()
_,axes=plt.subplots(figsize=(13,5))
bins=[2.3,5,10,15,20,30,40,60,90,120,150,180,220,260]
usr_item.hist(bins=bins,range=(0,250),histtype='bar');
plt.xticks(np.arange(0,250,10));
plt.xlabel('Article views count ');
plt.ylabel('User count')
plt.title('Distribution of the user-article interations');


# In[9]:

df.email.value_counts().sum()


# In[10]:

usr_item.describe()


# The statistical summmary reflects the histogram showing that 75% of the users interact with maximum of 9 articles,post which smaller number of users interact with higher number of articles.

# In[11]:

# Fill in the median and maximum number of user_article interactios below
median_val = 3 # 50% of individuals interact with 3 number of articles or fewer.
max_views_by_user = 364 # The maximum number of user-article interactions by any 1 user is 364.


# In[12]:

#No nulls,title accessed multiple times or interacted by multiple users
df.title.isnull().sum() , df.title.duplicated().sum()


# In[13]:

#there are articles which have been interacted with more than once by the same user
df[(df['email']=='0000b6387a0366322d7fbfc6434af145adf7fed1' )& (df['article_id']==43.0) ]


# `2.` Explore and remove duplicate articles from the **df_content** dataframe.  

# In[14]:

# Find and explore duplicate articles
df_content.head(2)


# In[15]:

#number of duplicate entries by doc name
df_content.doc_full_name.duplicated().sum()


# In[16]:

df_content[df_content.doc_full_name.duplicated()]


# In[17]:

pd.options.display.width = 0
pd.set_option('display.width', 1000)

#sample duplicate docs
df_content[df_content.doc_full_name=='Use the Primary Index']


# In[18]:

df_content[df_content.doc_full_name==df_content.iloc[761]['doc_full_name']]


# In[19]:

#Remove duplicate docs from df_content


# In[20]:

# Remove any rows that have the same article_id - only keep the first

df_content.article_id.duplicated().sum()


# In[21]:

#duplicate entries by article id and doc name are same
df_content[df_content.doc_full_name.duplicated()]==df_content[df_content.article_id.duplicated()]


# In[22]:

df_content=df_content.drop_duplicates('article_id')


# `3.` Use the cells below to find:
# 
# **a.** The number of unique articles that have an interaction with a user.  
# **b.** The number of unique articles in the dataset (whether they have any interactions or not).<br>
# **c.** The number of unique users in the dataset. (excluding null values) <br>
# **d.** The number of user-article interactions in the dataset.

# **a.**The number of unique articles that have an interaction with a user.

# In[23]:

df[df.email.notnull()].shape


# In[24]:

df[df.email.notnull()].article_id.nunique()


# **b**. The number of unique articles in the dataset (whether they have any interactions or not).

# In[25]:

df_content.article_id.nunique()


# **c**. The number of unique users in the dataset. (excluding null values) 

# In[26]:

len(df.email.value_counts()),df.email.nunique()


# **d**. The number of user-article interactions in the dataset.

# In[27]:

df.shape[0]


# In[28]:

unique_articles = 714 # The number of unique articles that have at least one interaction
total_articles = 1051 # The number of unique articles on the IBM platform
unique_users = 5148# The number of unique users
user_article_interactions = 45993 # The number of user-article interactions


# `4.` Use the cells below to find the most viewed **article_id**, as well as how often it was viewed.  After talking to the company leaders, the `email_mapper` function was deemed a reasonable way to map users to ids.  There were a small number of null values, and it was found that all of these null values likely belonged to a single user (which is how they are stored using the function below).

# In[29]:

df[df.email.notnull()]['article_id'].value_counts().head(5)


# In[30]:

most_viewed_article_id = '1429.0' # The most viewed article in the dataset as a string with one value following the decimal 
max_views = 937# The most viewed article in the dataset was viewed how many times?


# In[31]:

## No need to change the code here - this will be helpful for later parts of the notebook
# Run this cell to map the user email to a user_id column and remove the email column

def email_mapper():
    coded_dict = dict()
    cter = 1
    email_encoded = []
    
    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
        email_encoded.append(coded_dict[val])
    return email_encoded

email_encoded = email_mapper()
del df['email']
df['user_id'] = email_encoded

# show header
df.head()


# In[32]:

df.user_id.isnull().sum()


# In[33]:

df.article_id.describe()


# In[34]:

## If you stored all your results in the variable names above, 
## you shouldn't need to change anything in this cell

sol_1_dict = {
    '`50% of individuals have _____ or fewer interactions.`': median_val,
    '`The total number of user-article interactions in the dataset is ______.`': user_article_interactions,
    '`The maximum number of user-article interactions by any 1 user is ______.`': max_views_by_user,
    '`The most viewed article in the dataset was viewed _____ times.`': max_views,
    '`The article_id of the most viewed article is ______.`': most_viewed_article_id,
    '`The number of unique articles that have at least 1 rating ______.`': unique_articles,
    '`The number of unique users in the dataset is ______`': unique_users,
    '`The number of unique articles on the IBM platform`': total_articles
}

# Test your dictionary against the solution
t.sol_1_test(sol_1_dict)


# ### <a class="anchor" id="Rank">Part II: Rank-Based Recommendations</a>
# 
# Unlike in the earlier lessons, we don't actually have ratings for whether a user liked an article or not.  We only know that a user has interacted with an article.  In these cases, the popularity of an article can really only be based on how often an article was interacted with.
# 
# `1.` Fill in the function below to return the **n** top articles ordered with most interactions as the top. Test your function using the tests below.

# In[35]:

def get_top_articles(n, df=df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    
    '''
    # find the total unique values/views for the articles sorted in descending
    article_vws = df[['article_id','title']].article_id.value_counts().to_frame(name='views')
    article_vws=article_vws.reset_index().rename(columns={'index':'article_id'})
    article_vws = df[['article_id','title']].article_id.value_counts().to_frame(name='views')
    top_articles = article_vws.head(n)
    art_id=article_vws.head(n).index.tolist()
    top_articles=[]
    for id in art_id:
        
        title = df[df.article_id==id]['title'].values[0]
        top_articles.append(title)
    
    return top_articles # Return the top article titles from df (not df_content)

def get_top_article_ids(n, df=df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article ids 
    
    '''
    article_vws = df[['article_id','title']].article_id.value_counts().to_frame(name='views')
    article_vws=article_vws.reset_index().rename(columns={'index':'article_id'})
    article_vws = df[['article_id','title']].article_id.value_counts().to_frame(name='views')
    top_articles = article_vws.head(n)
    top_articles_id=article_vws.head(n).index.astype('str').tolist()
 
    return top_articles_id # Return the top article ids


# In[36]:

print('Top articles:\n-------------\n',get_top_articles(10),'\n')
print('Top article ids:\n----------------\n',get_top_article_ids(10))


# In[37]:

# Test your function by returning the top 5, 10, and 20 articles
top_5 = get_top_articles(5)
top_10 = get_top_articles(10)
top_20 = get_top_articles(20)

# Test each of your three lists from above
t.sol_2_test(get_top_articles)


# ### <a class="anchor" id="User-User">Part III: User-User Based Collaborative Filtering</a>
# 
# 
# `1.` Use the function below to reformat the **df** dataframe to be shaped with users as the rows and articles as the columns.  
# 
# * Each **user** should only appear in each **row** once.
# 
# 
# * Each **article** should only show up in one **column**.  
# 
# 
# * **If a user has interacted with an article, then place a 1 where the user-row meets for that article-column**.  It does not matter how many times a user has interacted with the article, all entries where a user has interacted with an article should be a 1.  
# 
# 
# * **If a user has not interacted with an item, then place a zero where the user-row meets for that article-column**. 
# 
# Use the tests to make sure the basic structure of your matrix matches what is expected by the solution.

# In[38]:

#Identify the unique number of interactions.There are 1 or more interactions .
df.groupby(['user_id','article_id'])['article_id'].count().to_frame(name='views')['views'].unique()


# In[39]:

# create the user-article matrix with 1's and 0's

def create_user_item_matrix(df):
    '''
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns
    
    OUTPUT:
    user_item - user item matrix 
    
    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with 
    an article and a 0 otherwise
    '''
    # Fill in the function here
    #create user item matrix
    user_item=df.groupby(['user_id','article_id'])['title'].count().unstack()
    
    for col in user_item.columns.tolist():
        user_item[col]=user_item[col].apply(lambda x: 1 if x >=1  else 0)   
    
    return user_item # return the user_item matrix 

user_item = create_user_item_matrix(df)


# In[40]:

user_item.head(2) 


# In[41]:

## Tests: You should just need to run this cell.  Don't change the code.
assert user_item.shape[0] == 5149, "Oops!  The number of users in the user-article matrix doesn't look right."
assert user_item.shape[1] == 714, "Oops!  The number of articles in the user-article matrix doesn't look right."
assert user_item.sum(axis=1)[1] == 36, "Oops!  The number of articles seen by user 1 doesn't look right."
print("You have passed our quick tests!  Please proceed!")


# `2.` Complete the function below which should take a user_id and provide an ordered list of the most similar users to that user (from most similar to least similar).  The returned result should not contain the provided user_id, as we know that each user is similar to him/herself. Because the results for each user here are binary, it (perhaps) makes sense to compute similarity as the dot product of two users. 
# 
# Use the tests to test your function.

# In[42]:

def find_similar_users(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user_id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    similar_users - (list) an ordered list where the closest users (largest dot product users)
                    are listed first
    
    Description:
    Computes the similarity of every pair of users based on the dot product
    Returns an ordered
    
    '''
    # compute similarity of each user to the provided user
    sim_user = user_item[user_item.index == user_id].dot(user_item.T)

    # sort by similarity
    similarity = sim_user.sort_values(user_id,axis=1,ascending=False)

    # create list of just the ids
    most_similar_users=similarity.columns.tolist()
    
    # remove the own user's id
    most_similar_users.remove(user_id)
       
    return most_similar_users # return a list of the users in order from most to least similar
        


# In[43]:

# Do a spot check of your function
print("The 10 most similar users to user 1 are: {}".format(find_similar_users(1)[:10]))
print("The 5 most similar users to user 3933 are: {}".format(find_similar_users(3933)[:5]))
print("The 3 most similar users to user 46 are: {}".format(find_similar_users(46)[:3]))


# `3.` Now that you have a function that provides the most similar users to each user, you will want to use these users to find articles you can recommend.  Complete the functions below to return the articles you would recommend to each user. 

# In[44]:

def get_article_names(article_ids, df=df):
    '''
    INPUT:
    article_ids - (list) a list of article ids
    df - (pandas dataframe) df as defined at the top of the notebook
    
    OUTPUT:
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the title column)
    '''
    # article names
    article_names = df[df['article_id'].isin(article_ids)]['title'].unique().tolist()
    
    return article_names # Return the article names associated with list of article ids


def get_user_articles(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the doc_full_name column in df_content)
    
    Description:
    Provides a list of the article_ids and article titles that have been seen by a user
    '''
    # get list of article_ids
    article_ids = user_item.loc[user_id][user_item.loc[user_id].values==1].index.tolist()
    article_ids = [str(ele) for ele in article_ids]

    #get the list of article names
    #article_names = df_content[df_content.article_id.isin(article_ids)]['doc_full_name'].tolist()
    article_names = get_article_names(article_ids)

    return article_ids, article_names # return the ids and names


def user_user_recs(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user
    
    OUTPUT:
    recs - (list) a list of recommendations for the user
    
    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Does this until m recommendations are found
    
    Notes:
    Users who are the same closeness are chosen arbitrarily as the 'next' user
    
    For the user where the number of recommended articles starts below m 
    and ends exceeding m, the last items are chosen arbitrarily
    
    '''
    # Get similar users like the user of interest(uoi)
    similar_users  = find_similar_users(user_id)
    
    #get the article ids and names viewed by  the uoi
    ids_vwd,titles_vwd = get_user_articles(user_id)
    
    #get the recommendations for the uoi
    recs = []
    #Find the similar users' article ids and names
    for usr in (find_similar_users(user_id)):
        ids,titles=get_user_articles(usr)
        
    #recs not seen by uoi
    rec_usr = np.setdiff1d(ids_vwd, ids, assume_unique=True)
    recs.append(list(rec_usr))
        
    recs = [item for nst_lst in recs for item in nst_lst]
    recs = recs[:m]
    
    return recs # return your recommendations for this user_id    


# In[45]:

# Check Results
get_article_names(user_user_recs(1, 10)) # Return 10 recommendations for user 1


# In[46]:

# Test your functions here - No need to change this code - just run this cell
assert set(get_article_names(['1024.0', '1176.0', '1305.0', '1314.0', '1422.0', '1427.0'])) == set(['using deep learning to reconstruct high-resolution audio', 'build a python app on the streaming analytics service', 'gosales transactions for naive bayes model', 'healthcare python streaming application demo', 'use r dataframes & ibm watson natural language understanding', 'use xgboost, scikit-learn & ibm watson machine learning apis']), "Oops! Your the get_article_names function doesn't work quite how we expect."
assert set(get_article_names(['1320.0', '232.0', '844.0'])) == set(['housing (2015): united states demographic measures','self-service data preparation with ibm data refinery','use the cloudant-spark connector in python notebook']), "Oops! Your the get_article_names function doesn't work quite how we expect."
assert set(get_user_articles(20)[0]) == set(['1320.0', '232.0', '844.0'])
assert set(get_user_articles(20)[1]) == set(['housing (2015): united states demographic measures', 'self-service data preparation with ibm data refinery','use the cloudant-spark connector in python notebook'])
assert set(get_user_articles(2)[0]) == set(['1024.0', '1176.0', '1305.0', '1314.0', '1422.0', '1427.0'])
assert set(get_user_articles(2)[1]) == set(['using deep learning to reconstruct high-resolution audio', 'build a python app on the streaming analytics service', 'gosales transactions for naive bayes model', 'healthcare python streaming application demo', 'use r dataframes & ibm watson natural language understanding', 'use xgboost, scikit-learn & ibm watson machine learning apis'])
print("If this is all you see, you passed all of our tests!  Nice job!")


# `4.` Now we are going to improve the consistency of the **user_user_recs** function from above.  
# 
# * Instead of arbitrarily choosing when we obtain users who are all the same closeness to a given user - choose the users that have the most total article interactions before choosing those with fewer article interactions.
# 
# 
# * Instead of arbitrarily choosing articles from the user where the number of recommended articles starts below m and ends exceeding m, choose articles with the articles with the most total interactions before choosing those with fewer total interactions. This ranking should be  what would be obtained from the **top_articles** function you wrote earlier.

# In[47]:

def get_top_sorted_users(user_id, df=df, user_item=user_item):
    '''
    INPUT:
    user_id - (int)
    df - (pandas dataframe) df as defined at the top of the notebook 
    user_item - (pandas dataframe) matrix of users by articles: 
            1's when a user has interacted with an article, 0 otherwise
    
            
    OUTPUT:
    neighbors_df - (pandas dataframe) a dataframe with:
                    neighbor_id - is a neighbor user_id
                    similarity - measure of the similarity of each user to the provided user_id
                    num_interactions - the number of articles viewed by the user - if a u
                    
    Other Details - sort the neighbors_df by the similarity and then by number of interactions where 
                    highest of each is higher in the dataframe
     
    '''
    # usr_item interations
    usr_item_interactions = df.groupby(['user_id'])['article_id'].count()
    
    #Get the total number of users
    tot_users = user_item.shape[0]
    
    #Get the  similar users interactions with the input user_id
    similarity =[]
    num_interactions = []
    
    #list of all neighbor users
    neighbor_id = [id for id in range(1, tot_users) if id != user_id]
    
    #for the given user_id find the neighbour user's similarity measure and number of interactions
    for n_id in neighbor_id:
        similarity.append(np.dot(user_item.loc[user_id], user_item.loc[n_id]))
        num_interactions.append(usr_item_interactions.loc[n_id])
    
    #Create the neighbors_df
    neighbors_df =pd.DataFrame({'neighbor_id':neighbor_id,'similarity':similarity,'num_interactions':num_interactions})
    
    #sort the neighbors_df by the similarity and then by number of interactions
    neighbors_df = neighbors_df.sort_values(['similarity','num_interactions'],ascending=False)
    
    #sort the neighbors_df by the similarity and then by number of interactions
    neighbors_df.sort_values(['similarity','num_interactions'],ascending=[False,False])
    
    return neighbors_df # Return the dataframe specified in the doc_string


def user_user_recs_part2(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user
    
    OUTPUT:
    recs - (list) a list of recommendations for the user by article id
    rec_names - (list) a list of recommendations for the user by article title
    
    Description:
    Loops through the users based on closeness to the input user_id
    For each user - finds articles the user hasn't seen before and provides them as recs
    Do this until m recommendations are found
    
    Notes:
    * Choose the users that have the most total article interactions 
    before choosing those with fewer article interactions.

    * Choose articles with the articles with the most total interactions 
    before choosing those with fewer total interactions. 
   
    '''
    # users most similar and with most total article interactions
    neighbors_df = get_top_sorted_users(user_id)
        
    # get top-m neighbor_id
    top_m_neighbors = list(neighbors_df[:m]['neighbor_id'])

    # set article_ids seen by top-m neighbors
    for n_id in top_m_neighbors:
        recs = user_item.loc[n_id][user_item.loc[n_id].values==1].index.unique().astype('str').tolist()[:m]

    # article names 
    rec_names = list(set(df[df['article_id'].isin(recs)]['title']))
    
    return recs, rec_names


# In[48]:

# Quick spot check - don't change this code - just use it to test your functions
rec_ids, rec_names = user_user_recs_part2(20, 10)
print("The top 10 recommendations for user 20 are the following article ids:")
print(rec_ids)
print()
print("The top 10 recommendations for user 20 are the following article names:")
print(rec_names)


# `5.` Use your functions from above to correctly fill in the solutions to the dictionary below.  Then test your dictionary against the solution.  Provide the code you need to answer each following the comments below.

# In[49]:

### Tests with a dictionary of results

user1_most_sim = get_top_sorted_users(1)['neighbor_id'].iloc[0]# Find the user that is most similar to user 1 
user131_10th_sim = get_top_sorted_users(131)['neighbor_id'].iloc[9]# Find the 10th most similar user to user 131


# In[50]:

get_top_sorted_users(1)['neighbor_id'].iloc[0],get_top_sorted_users(131)['neighbor_id'].iloc[9]


# In[51]:

## Dictionary Test Here
sol_5_dict = {
    'The user that is most similar to user 1.': user1_most_sim, 
    'The user that is the 10th most similar to user 131': user131_10th_sim,
}

t.sol_5_test(sol_5_dict)


# `6.` If we were given a new user, which of the above functions would you be able to use to make recommendations?  Explain.  Can you think of a better way we might make recommendations?  Use the cell below to explain a better method for new users.

# **The recommendations made thus far are based on measure of similarity ie. user-item interactions and the number of interactions.New recommendations made for specific user based on these parameters for the most similar neighbouring users.However new users would have fewer to nil user-item interactions. In such a scenario recommendations based on popularity /top articles can be made.**
# 
# **We shall use get_top_article_ids function to recommend top articles for the new users. For new users and new items rank based and content based recommendations can be used for making recommendations**

# `7.` Using your existing functions, provide the top 10 recommended articles you would provide for the a new user below.  You can test your function against our thoughts to make sure we are all on the same page with how we might make a recommendation.

# In[52]:

new_user = '0.0'

# What would your recommendations be for this new user '0.0'?  As a new user, they have no observed articles.
# Provide a list of the top 10 article ids you would give to 
new_user_recs = get_top_article_ids(10)# Your recommendations here
new_user_recs


# In[53]:

assert set(new_user_recs) == set(['1314.0','1429.0','1293.0','1427.0','1162.0','1364.0','1304.0','1170.0','1431.0','1330.0']), "Oops!  It makes sense that in this case we would want to recommend the most popular articles, because we don't know anything about these users."

print("That's right!  Nice job!")


# ### <a class="anchor" id="Content-Recs">Part IV: Content Based Recommendations (EXTRA - NOT REQUIRED)</a>
# 
# Another method we might use to make recommendations is to perform a ranking of the highest ranked articles associated with some term.  You might consider content to be the **doc_body**, **doc_description**, or **doc_full_name**.  There isn't one way to create a content based recommendation, especially considering that each of these columns hold content related information.  
# 
# `1.` Use the function body below to create a content based recommender.  Since there isn't one right answer for this recommendation tactic, no test functions are provided.  Feel free to change the function inputs if you decide you want to try a method that requires more input values.  The input values are currently set with one idea in mind that you may use to make content based recommendations.  One additional idea is that you might want to choose the most popular recommendations that meet your 'content criteria', but again, there is a lot of flexibility in how you might make these recommendations.
# 
# ### This part is NOT REQUIRED to pass this project.  However, you may choose to take this on as an extra way to show off your skills.

# In[54]:

def make_content_recs():
    '''
    INPUT:
    
    OUTPUT:
    
    '''


# `2.` Now that you have put together your content-based recommendation system, use the cell below to write a summary explaining how your content based recommender works.  Do you see any possible improvements that could be made to your function?  Is there anything novel about your content based recommender?
# 
# ### This part is NOT REQUIRED to pass this project.  However, you may choose to take this on as an extra way to show off your skills.

# **Write an explanation of your content based recommendation system here.**

# `3.` Use your content-recommendation system to make recommendations for the below scenarios based on the comments.  Again no tests are provided here, because there isn't one right answer that could be used to find these content based recommendations.
# 
# ### This part is NOT REQUIRED to pass this project.  However, you may choose to take this on as an extra way to show off your skills.

# In[55]:

# make recommendations for a brand new user

# make a recommendations for a user who only has interacted with article id '1427.0'


# ### <a class="anchor" id="Matrix-Fact">Part V: Matrix Factorization</a>
# 
# In this part of the notebook, you will build use matrix factorization to make article recommendations to the users on the IBM Watson Studio platform.
# 
# `1.` You should have already created a **user_item** matrix above in **question 1** of **Part III** above.  This first question here will just require that you run the cells to get things set up for the rest of **Part V** of the notebook. 

# In[56]:

# Load the matrix here
user_item_matrix = pd.read_pickle('user_item_matrix.p')


# In[57]:

# quick look at the matrix
user_item_matrix.head()


# `2.` In this situation, you can use Singular Value Decomposition from [numpy](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.linalg.svd.html) on the user-item matrix.  Use the cell to perform SVD, and explain why this is different than in the lesson.

# In[58]:

# Perform SVD on the User-Item Matrix Here

u, s, vt = np.linalg.svd(user_item_matrix, full_matrices=False) # use the built in to get the three matrices


# In[59]:

u.shape, s.shape, vt.shape


# **Application of SVD is possible as there are no missing values. There are only two values -interaction with article,1  and no interaction with the article,0. However in the lesson, presence of nan values broke the code. This is resolved by implementing FunkSVD**
# 
# **Besides the missing nan values, the sigma matrix has an array of 714 elements or 714 latent factors unlike the lesson which had only 4 latent features. **

# `3.` Now for the tricky part, how do we choose the number of latent features to use?  Running the below cell, you can see that as the number of latent features increases, we obtain a lower error rate on making predictions for the 1 and 0 values in the user-item matrix.  Run the cell below to get an idea of how the accuracy improves as we increase the number of latent features.

# In[60]:

num_latent_feats = np.arange(10,700+10,20)
sum_errs = []

for k in num_latent_feats:
    # restructure with k latent features
    s_new, u_new, vt_new = np.diag(s[:k]), u[:, :k], vt[:k, :]
    
    # take dot product
    user_item_est = np.around(np.dot(np.dot(u_new, s_new), vt_new))
    
    # compute error for each prediction to actual value
    diffs = np.subtract(user_item_matrix, user_item_est)
    
    # total errors and keep track of them
    err = np.sum(np.sum(np.abs(diffs)))
    sum_errs.append(err)
    
    
plt.plot(num_latent_feats, 1 - np.array(sum_errs)/df.shape[0]);
plt.xlabel('Number of Latent Features');
plt.ylabel('Accuracy');
plt.title('Accuracy vs. Number of Latent Features');


# `4.` From the above, we can't really be sure how many features to use, because simply having a better way to predict the 1's and 0's of the matrix doesn't exactly give us an indication of if we are able to make good recommendations.  Instead, we might split our dataset into a training and test set of data, as shown in the cell below.  
# 
# Use the code from question 3 to understand the impact on accuracy of the training and test sets of data with different numbers of latent features. Using the split below: 
# 
# * How many users can we make predictions for in the test set?  
# * How many users are we not able to make predictions for because of the cold start problem?
# * How many articles can we make predictions for in the test set?  
# * How many articles are we not able to make predictions for because of the cold start problem?

# In[61]:

df_train = df.head(40000)
df_test = df.tail(5993)

def create_test_and_train_user_item(df_train, df_test):
    '''
    INPUT:
    df_train - training dataframe
    df_test - test dataframe
    
    OUTPUT:
    user_item_train - a user-item matrix of the training dataframe 
                      (unique users for each row and unique articles for each column)
    user_item_test - a user-item matrix of the testing dataframe 
                    (unique users for each row and unique articles for each column)
    test_idx - all of the test user ids
    test_arts - all of the test article ids
    
    '''
    # create user-item matrix of the training and test dataframe
    user_item_train = create_user_item_matrix(df_train)
    user_item_test = create_user_item_matrix(df_test)
    
    test_idx = set(user_item_test.index)   
    test_arts =  set(user_item_test.columns)

    return user_item_train, user_item_test, test_idx, test_arts

user_item_train, user_item_test, test_idx, test_arts = create_test_and_train_user_item(df_train, df_test)


# In[62]:

#How many users can we make predictions for in the test set?
user_item_test.head(2)


# In[63]:

len(user_item_train.index.unique()),len(user_item_test.index.unique())


# In[64]:

len(np.intersect1d(user_item_train.index.unique(),user_item_test.index.unique()))


# In[65]:

#How many users in the test set are we not able to make predictions for because of the cold start problem?
len(set(user_item_test.index) - set(user_item_train.index))


# In[66]:

#How many articles can we make predictions for in the test set?
len(np.intersect1d(user_item_train.columns.unique(),user_item_test.columns.unique()))


# In[67]:

#How many articles in the test set are we not able to make predictions for because of the cold start problem?
len(set(user_item_test.columns) - set(user_item_train.columns))


# In[68]:

# Replace the values in the dictionary below
a = 662 
b = 574 
c = 20 
d = 0 


sol_4_dict = {
    'How many users can we make predictions for in the test set?': c, 
    'How many users in the test set are we not able to make predictions for because of the cold start problem?': a, 
    'How many movies can we make predictions for in the test set?': b,
    'How many movies in the test set are we not able to make predictions for because of the cold start problem?': d
}

t.sol_4_test(sol_4_dict)



# `5.` Now use the **user_item_train** dataset from above to find U, S, and V transpose using SVD. Then find the subset of rows in the **user_item_test** dataset that you can predict using this matrix decomposition with different numbers of latent features to see how many features makes sense to keep based on the accuracy on the test data. This will require combining what was done in questions `2` - `4`.
# 
# Use the cells below to explore how well SVD works towards making predictions for recommendations on the test data.  

# In[91]:

# fit SVD on the user_item_train matrix
u_train, s_train, vt_train = np.linalg.svd(np.array(user_item_train, dtype='int'), full_matrices=False)# fit svd similar to above then use the cells below


# In[92]:

u_train.shape, s_train.shape, vt_train.shape


# In[71]:

# Use these cells to see how well you can use the training 
# decomposition to predict on test data


# In[93]:

#Get the train items and users
train_idx = user_item_train.index
train_arts = user_item_train.columns


# In[94]:

#Common users and articles in train and test data
com_idx = train_idx.isin(test_idx)
com_arts = train_arts.isin(test_arts)


# In[95]:

#find subset of user and article matrices
u_test = u_train[com_idx,:]
vt_test = vt_train[:,com_arts]


# In[96]:

u_test.shape,vt_test.shape


# In[101]:

#Subset of users and articles from user_item_test
test_users = set(train_idx) & set(test_idx)
test_articles = set(train_arts) & set(test_arts)
user_item_test_pred = user_item_test.loc[test_users, test_articles]

len(test_users),len(test_articles),user_item_test_pred.shape


# In[106]:

num_latent_feats = np.arange(10,700+10,20)
sum_errs_train = []
sum_errs_test = []


for k in num_latent_feats:
    # restructure with k latent features
    s_new_train, u_new_train, vt_new_train = np.diag(s_train[:k]), u_train[:, :k], vt_train[:k, :]
    
    u_new_test,vt_new_test = u_test[:, :k],vt_test[:k, :]
    
    
    # take dot product
    user_item_train_est = np.around(np.dot(np.dot(u_new_train, s_new_train), vt_new_train))
    
    user_item_test_est = np.around(np.dot(np.dot(u_new_test, s_new_train), vt_new_test))
    
    # compute error for each prediction to actual value
    train_diffs = np.subtract(user_item_train, user_item_train_est)
    test_diffs = np.subtract(user_item_test_pred, user_item_test_est)
    
    # total errors and keep track of them
    err_train = np.sum(np.sum(np.abs(train_diffs)))
    err_test = np.sum(np.sum(np.abs(test_diffs)))
    
    sum_errs_train.append(err_train)
    sum_errs_test.append(err_test)
    


# In[154]:

#Plot Accuracy vs. Number of Latent Features
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(num_latent_feats, 1 - np.array(sum_errs_train)/((df.shape[0])), label="Train accuracy")
ax2.plot(num_latent_feats, 1 - np.array(sum_errs_test)/((df.shape[0])), color='orange', label="Test accuracy")

ax1.legend(loc=1, bbox_to_anchor=(0.9,0.9))
ax1.set_xlabel('Number of Latent Features')
ax1.set_ylabel('Train accuracy')
ax1.set_title('Accuracy vs. Number of Latent Features')
ax2.set_ylabel('Test accuracy')
ax2.legend(loc=1, bbox_to_anchor=(0.9,0.2))
plt.show()


# In[150]:

#(user_item_train.shape[0]*user_item_train.shape[1]),df.shape[0],(user_item_test.shape[0]*user_item_test.shape[1])


# `6.` Use the cell below to comment on the results you found in the previous question. Given the circumstances of your results, discuss what you might do to determine if the recommendations you make with any of the above recommendation systems are an improvement to how users currently find articles? 

# **Accuracy for train data increases with latent features stabilizing over 500 latent features. The exact opposite is what we  observe for the test data. There is a steep fall in the accuracy  post few latent features(31-32). The drastic difference is contributed by few common users between train and test data ie predictions can be made for only 20 users and rest fails due to cold start problem.**
# 
# **Rank based recommendations for new users is suited for cold start scenario and content based recommendation for new movies.**
# 
# **In order to know how well the recommendations work, these recommendations have to deployed to users and  how these translates to metrics like clicks,conversions,etc. Performance evaluation of these recommendations are through experimental,nonexperimental and simulated methods.** 
# 
# **Experimental methods like online testing involves  AB testing, comparing the 'old' vs 'new' version of recommended items.
# The non-experimental method of evaluation can be done via offline testing where the recommendations are evaluated for how good they are by rating the recommendations.**
# 

# In[1]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Recommendations_with_IBM.ipynb'])

