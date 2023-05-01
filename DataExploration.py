#!/usr/bin/env python
# coding: utf-8

# ## Load the dataset

# In[184]:


import json
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import scipy

with open('./data/recipes_with_nutritional_info.json') as f:
    dataset = json.load(f)


# ## Data Analysis
# 
# We examine our dataset to determine the most commonly used ingredients, and actions within recipe instructions. We do this so we can better inform our choices for the ingredients and actions we will consider in our Markov Model

# In[2]:


keys = list(dataset[0].keys())

ingredients = list()
# print(dataset[0]['ingredients'])

for entry in dataset:
    ingredient_list = [item['text'].split(',')[0] for item in entry['ingredients']]
    ingredients.extend(ingredient_list)
    
print(len(list(set(ingredients)))) # Get the number of unique ingredients

ingredient_freq = {i: ingredients.count(i) for i in set(ingredients)}
sorted_ingredient_freq = sorted(ingredient_freq.items(), key=lambda x: x[1])


# In[3]:


actions = list()

for item in dataset:
    for line in item['instructions']:
        tokens = line['text'].split(' ')
        actions.append(tokens[0])
        
action_freq = {i: actions.count(i) for i in set(actions)}
sorted_action_freq = sorted(action_freq.items(), key=lambda x: x[1])


# In[4]:


print(sorted_ingredient_freq[-20:-1])
print(sorted_action_freq[-20:-1])


# ## Setting up the model
# 
# Looking over the data we just collected, we can make a judicious choice of what ingredients and actions we should consider for our model. Ideally, the ingredients should be very commonly used. Actions should be sensible (they shouldn't, for example, be words like 'In'). They should also ideally 'act' on more than one ingredient at a time. Our hyperedges won't be meaningful if they just consist of one-to-one connections

# In[5]:


from copy import deepcopy

# For now, let us just take a fixed number of the most common ingredients and actions
NUM_INGREDIENTS = 20
NUM_ACTIONS = 20

ingredients = [item[0] for item in sorted_ingredient_freq[:-NUM_INGREDIENTS:-1]]
actions = [item[0] for item in sorted_action_freq[:-NUM_ACTIONS:-1]]


# In[6]:


# Now, we convert all instructions in the dataset into a format where we can extract the 
# conditional frequencies

feature_space = []
for entry in dataset:
    recipe_instructions = []
    
    for line in entry['instructions']:
        tokens = line['text'].split(' ')
        
        if tokens[0] not in actions:
            continue
        
        features = [tokens[0]] + list(filter(lambda s: s in ingredients, tokens))
        actions.append(tokens[0])
        recipe_instructions.append(features)
        
    feature_space.append(deepcopy(recipe_instructions))


# In[32]:


import time

conditional_freq = {i: [] for i in actions}
action_freq = {i: [] for i in actions}

for entry in feature_space:
    ingredient_combinations = [line[1:] for line in entry if len(line) > 1]
    action_list = [line[0] for line in entry if len(line) != 0]
    
    if len(ingredient_combinations) == 0:
        continue
    
    for i in range(len(entry)):
        action_freq[entry[i][0]].append(action_list[:i+1])
        
        if entry[i][0] in actions:
            up_to = min(i+1, len(ingredient_combinations)) # Sometimes there are more actions than ingredients
            conditional_freq[entry[i][0]].extend(ingredient_combinations[:up_to])
 
# Reduce to frequency table
for key in conditional_freq:
    conditional_freq[key] = {tuple(i): conditional_freq[key].count(i) for i in np.unique(conditional_freq[key])}
    action_freq[key] = {tuple(i): action_freq[key].count(i) for i in np.unique(action_freq[key])}


# In[182]:


import pickle

## Save to disk since this shit takes forever to compute
with open('data/20x20.pkl', 'wb') as f:
    pickle.dump([conditional_freq, action_freq], f)

