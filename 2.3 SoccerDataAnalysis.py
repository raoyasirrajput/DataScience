#!/usr/bin/env python
# coding: utf-8

# <br><br><center><h1 style="font-size:4em;color:#2467C0">Week 1: Soccer Data Analysis</h1></center>
# <br>
# <table>
# <col width="550">
# <col width="450">
# <tr>
# <td><img src="https://images.pexels.com/photos/1667583/pexels-photo-1667583.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=750&w=1260" align="middle" style="width:550px;height:360px;"/></td>
# <td>
# This week, we will be using an open dataset from the popular site <a href="https://www.kaggle.com">Kaggle</a>. This <a href="https://www.kaggle.com/hugomathien/soccer">European Soccer Database</a> has more than 25,000 matches and more than 10,000 players for European professional soccer seasons from 2008 to 2016. 
# <br>
# <br>
# Although we won’t be getting into the details of it for our example, the dataset even has attributes on weekly game updates, team line up, and detailed match events.
# <br>
# <br>
# The goal of this notebook is to walk you through an end to end process of analyzing a dataset and introduce you to what we will be covering in this course. Our simple analytical process will include some steps for exploring  and cleaning our dataset, some steps for predicting player performance using basic statistics, and some steps for grouping similar clusters using machine learning. 
# <br>
# <br>
# Let's get started with our Python journey!
# </td>
# </tr>
# </table>

# ## Getting Started
# <br> To get started, we will need to:
# <ol>
# <li>Download the data from: <a href="https://www.kaggle.com/hugomathien/soccer">https://www.kaggle.com/hugomathien/soccer</a></li>
# <li>Extract the zip file called "soccer.zip"</li>
#     <li>Move the extracted file `database.sqlite` to your Week 1 folder</li>
# </ol>

# ## Import Libraries
# <br> We will start by importing the Python libraries we will be using in this analysis. These libraries include:
# <ul>
# <li><b>sqllite3</b> for interacting with a local relational database; and</li>
# <li><b>pandas</b> and <b>numpy</b> for data ingestion and manipulation.</li>
# <li><b>matplotlib</b> for data visualization</li>
# <li>specific methods from <b>sklearn</b> for Machine Learning and 
# <li><b>customplot</b>, which contains custom functions we have written for this notebook</li>
# 
# </ul>

# In[ ]:


import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from customplot import *


# #### Ingest Data
# 
# Now, we will need to read the dataset using the commands below. 
# 
# <b>Note:</b> Make sure you run the import cell above (shift+enter) before you run the data ingest code below.
# 
# <b>df</b> is a variable pointing to a pandas data frame. We will learn about them in an upcoming week.

# In[ ]:


# Create your connection.
cnx = sqlite3.connect('database.sqlite')
df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)


# <h1 style="font-size:2em;color:#2467C0">Exploring Data</h1>
# 
# We will start our data exploration by generating simple statistics of the data. 
# <br><br> 
# Let us look at what the data columns are using a pandas attribute called "columns".

# In[ ]:


df.columns


# Next will display simple statistics of our dataset. You need to run each cell to make sure you see the outputs.

# In[ ]:


df.describe().transpose()


# <h1 style="font-size:2em;color:#2467C0">Data Cleaning: Handling Missing Data</h1>
# Real data is never clean. We need to make sure we clean the data by converting or getting rid of null or missing values.<br>
# The next code cell will show you if any of the 183978 rows have null value in one of the 42 columns.

# In[ ]:


#is any row NULL ?
df.isnull().any().any(), df.shape


# Now let's try to find how many data points in each column are null.

# In[ ]:


df.isnull().sum(axis=0)


# ## Fixing Null Values by Deleting Them
# 
# In our next two lines, we will drop the null values by going through each row.
# 

# In[ ]:


# Fix it

# Take initial # of rows
rows = df.shape[0]

# Drop the NULL rows
df = df.dropna()


# Now if we check the null values and number of rows, we will see that there are no null values and number of rows decreased accordingly.

# In[ ]:


#Check if all NULLS are gone ?
print(rows)
df.isnull().any().any(), df.shape


# To find exactly how many lines we removed, we need to subtract the current number of rows in our data frame from the original number of rows.

# In[ ]:


#How many rows with NULL values?

rows - df.shape[0]


# Our data table has many lines as you have seen. We can only look at few lines at once. Instead of looking at same top 10 lines every time, we shuffle - so we get to see different random sample on top. This way, we make sure the data is not in any particular order when we try sampling from it (like taking top or bottom few rows) by randomly shuffling the rows.

# In[ ]:


#Shuffle the rows of df so we get a distributed sample when we display top few rows

df = df.reindex(np.random.permutation(df.index))


# <h1 style="font-size:2em;color:#2467C0">Predicting: 'overall_rating' of a player</h1>
# Now that our data cleaning step is reasonably complete and we can trust and understand the data more, we will start diving into the dataset further. 

# ### Let's take a look at top few rows.
# 
# We will use the head function for data frames for this task. This gives us every column in every row.

# In[ ]:


df.head(5)


# Most of the time, we are only interested in plotting some columns. In that case, we can use the pandas column selection option as follows. Please ignore the first column in the output of the one line code below. It is the unique identifier that acts as an index for the data.<br><br>
# <b>Note:</b> From this point on, we will start referring to the columns as "features" in our description.

# In[ ]:


df[:10][['penalties', 'overall_rating']]


# ## Feature Correlation Analysis 
# Next, we will check if 'penalties' is correlated to 'overall_rating'. We are using a similar selection operation, bu this time for all the rows and within the correlation function. 

# # Are these correlated (using Pearson's correlation coefficient)?

# In[ ]:


df['overall_rating'].corr(df['penalties'])


# We see that Pearson's Correlation Coefficient for these two columns is 0.39. <br><br>
# Pearson goes from -1 to +1. A value of 0 would have told there is no correlation, so we shouldn’t bother looking at that attribute. A value of 0.39 shows some correlation, although it could be stronger. <br><br>
# At least, we have these attributes which are slightly correlated. This gives us hope that we might be able to build a meaningful predictor using these ‘weakly’ correlated features.<br><br>
# Next, we will create a list of features that we would like to iterate the same operation on.

# ## Create a list of potential Features that you want to measure correlation with

# In[ ]:


potentialFeatures = ['acceleration', 'curve', 'free_kick_accuracy', 'ball_control', 'shot_power', 'stamina']


# The for loop below prints out the correlation coefficient of "overall_rating" of a player with each feature we added to the list as potential.

# In[ ]:


# check how the features are correlated with the overall ratings

for f in potentialFeatures:
    related = df['overall_rating'].corr(df[f])
    print("%s: %f" % (f,related))


# ## Which features have the highest correlation with overall_rating?
# 
# Looking at the values printed by the previous cell, we notice that the to two are "ball_control" (0.44) and "shot_power" (0.43). So these two features seem to have higher correlation with "overall_rating".
# 

# <h1 style="font-size:2em;color:#2467C0">Data Visualization:</h1>
# Next we will start plotting the correlation coefficients of each feature with "overall_rating". We start by selecting the columns and creating a list with correlation coefficients, called "correlations".

# In[ ]:


cols = ['potential',  'crossing', 'finishing', 'heading_accuracy',
       'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',
       'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
       'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
       'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',
       'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',
       'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',
       'gk_reflexes']


# In[ ]:


# create a list containing Pearson's correlation between 'overall_rating' with each column in cols
correlations = [ df['overall_rating'].corr(df[f]) for f in cols ]


# In[ ]:


len(cols), len(correlations)


# We make sure that the number of selected features and the correlations calculated are the same, e.g., both 34 in this case. Next couple of cells show some lines of code that use pandas plaotting functions to create a 2D graph of these correlation vealues and column names. 

# In[ ]:


# create a function for plotting a dataframe with string columns and numeric values

def plot_dataframe(df, y_label):  
    color='coral'
    fig = plt.gcf()
    fig.set_size_inches(20, 12)
    plt.ylabel(y_label)

    ax = df.correlation.plot(linewidth=3.3, color=color)
    ax.set_xticks(df.index)
    ax.set_xticklabels(df.attributes, rotation=75); #Notice the ; (remove it and see what happens !)
    plt.show()


# In[ ]:


# create a dataframe using cols and correlations

df2 = pd.DataFrame({'attributes': cols, 'correlation': correlations}) 


# In[ ]:


# let's plot above dataframe using the function we created
    
plot_dataframe(df2, 'Player\'s Overall Rating')


# <h1 style="font-size:1.5em;color:#FB41C4">Analysis of Findings</h1>
# 
# Now it is time for you to analyze what we plotted. Suppose you have to predict a player's overall rating. Which 5 player attributes would you ask for?
# <br><br>
# <b>Hint:</b> Which are the five features with highest correlation coefficients?

# <h1 style="font-size:2em;color:#2467C0">Clustering Players into Similar Groups</h1>
# 
# Until now, we used basic statistics and correlation coefficients to start forming an opinion, but can we do better? What if we took some features and start looking at each player using those features? Can we group similar players based on these features? Let's see how we can do this. 
# 
# <b>Note:</b> Generally, someone with domain knowledge needs to define which features. We could have also selected some of the features with highest correlation with overall_rating. However, it does not guarantee best outcome always as we are not sure if the top five features are independent. For example, if 4 of the 5 features depend on the remaining 1 feature, taking all 5 does not give new information.

# ## Select Features on Which to Group Players

# In[ ]:


# Define the features you want to use for grouping players

select5features = ['gk_kicking', 'potential', 'marking', 'interceptions', 'standing_tackle']
select5features


# In[ ]:


# Generate a new dataframe by selecting the features you just defined

df_select = df[select5features].copy(deep=True)


# In[ ]:


df_select.head()


# ## Perform KMeans Clustering
# 
# Now we will use a machine learning method called KMeans to cluster the values (i.e., player features on `gk_kicking`, `potential`, `marking`, `interceptions`, and `standing_tackle`). We will ask for four clusters. We will talk about KMeans clustering and other machine learning tools in Python in Week 7 so we won't discuss these methods here.

# In[ ]:


# Perform scaling on the dataframe containing the features

data = scale(df_select)

# Define number of clusters
noOfClusters = 4

# Train a model
model = KMeans(init='k-means++', n_clusters=noOfClusters, n_init=20).fit(data)


# In[ ]:


print(90*'_')
print("\nCount of players in each cluster")
print(90*'_')

pd.value_counts(model.labels_, sort=False)


# If you find that the below cell runs into an error, make sure you are running the latest version of `pandas` on your computer (try `pip install pandas` in your terminal). Save and restart your Jupyter session once you have updated `pandas`.

# In[ ]:


# Create a composite dataframe for plotting
# ... Use custom function declared in customplot.py (which we imported at the beginning of this notebook)

P = pd_centers(featuresUsed=select5features, centers=model.cluster_centers_)
P


# <h1 style="font-size:2em;color:#2467C0">Visualization of Clusters</h1>
# We now have 4 clusters based on the features we selected, we can treat them as profiles for similar groups of players. We can visualize these profiles by plotting the centers for each cluster, i.e., the average values for each featuere within the cluster. We will use matplotlib for this visualization. We will learn more about matplotlib in Week 5. 

# In[ ]:


# For plotting the graph inside the notebook itself, we use the following command

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


parallel_plot(P)


# <h1 style="font-size:1.5em;color:#FB41C4">Analysis of Findings</h1>
# ### Can you identify the groups for each of the below?
# 
# <ul>
# <li>Two groups are very similar except in `gk_kicking` - these players can coach each other on `gk_kicking`, where they differ.</li>
# <li>Two groups are somewhat similar to each other except in potential.</li>
# </ul>

# In[ ]:




