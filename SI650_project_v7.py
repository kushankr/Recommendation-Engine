import numpy as np
import json
import math
from scipy.stats import pearsonr
import timeit
import random
import scipy.sparse as sps
from  nltk.corpus import stopwords
import string
from collections import OrderedDict
from sklearn.neighbors import DistanceMetric
#Source: Programming collective intelligence: Toby Segaran
#Source: https://www.kaggle.com/c/yelp-recsys-2013
#Source: RecSys Challenge 2013: Yelp business rating prediction 
#Source: Programming collective intelligence: Toby Segaran
fhand = open('yelp_training_set_review.json','rU');
details_dict={};
for line in fhand:	
    data_string=json.loads(line);
    if data_string['user_id'] not in details_dict:
    	details_dict[data_string['user_id']] = {};
    details_dict[data_string['user_id']][data_string['business_id']] = data_string['stars'];

dist_arr = ["euclidean","manhattan","chebyshev","minkowski"];
file_write = open("si650_project_v7.txt","w");
for it in dist_arr:
	dist = DistanceMetric.get_metric(it)
	import itertools
	#Source: http://stackoverflow.com/questions/12988351/split-a-dictionary-into-2-dictionaries
	def splitDict(d):
	    n = len(d) // 2        
	    i = iter(d.items())
	    d1 = dict(itertools.islice(i, n))   # grab first n items
	    d2 = dict(i)                        # grab the rest

	    return d1, d2

	dict_train={};
	dict_test={};
	for key,value in details_dict.iteritems():
		if len(value) >= 2:
			dict_train[key],dict_test[key] = splitDict(value); 
		else: 
			dict_train[key] = value;



	#print details_dict.items()[0];
	#print dict_train.items()[0];
	#print dict_test.items()[0];

	print len(dict_test)
	def sim_pearson(prefs,p1,p2):
	# Get the list of mutually rated items
	 #print p1, p2;
	 documents=[];
	 si={}
	 for item in prefs[p1]:
	  if item in prefs[p2]: si[item]=1
	# Find the number of elements
	 n=len(si)
	 #print n;
	# if they are no ratings in common, return 0
	 if n==0: return 0
	 #print len(documents)
	 #tfidf = vectorizer.fit_transform(documents)
	 #similarity = (tfidf*tfidf.T).A
	 #print similarity 
	 #return similarity[0][1];
	# Add up all the preferences
	 sum1=sum([prefs[p1][it] for it in si])
	 sum2=sum([prefs[p2][it] for it in si])
	 documents.append([prefs[p1][it] for it in si])
	 documents.append([prefs[p2][it] for it in si]);
	 d =  dist.pairwise(documents)[0][1];
	 r = 1/1+d;
	 return r;  

	#prefs = dict_train;
	#for k in dict_test.keys():
	 #person = k;
	 #scores=[(sim_pearson(prefs,person,other),other)
	 #for other in prefs if other!=person]
	# Sort the list so the highest scores appear at the top
	 #scores.sort( )
	 #scores.reverse( )
	    
	# Gets recommendations for a person by using a weighted average
	# of every other user's rankings
	def getRecommendations(prefs,person,similarity=sim_pearson):
	 totals={}
	 simSums={}
	 for other in prefs:
	# don't compare me to myself
	  if other==person: continue

	  sim=similarity(prefs,person,other)
	  #print sim
	# ignore scores of zero or lower
	  if sim<=0: continue
	  for item in prefs[other]:
	# only score movies I haven't seen yet
	   #if item not in prefs[person] or prefs[person][item]==0:
	# Similarity * Score
	   totals.setdefault(item,0)
	   totals[item]+=prefs[other][item]*sim
	# Sum of similarities
	   simSums.setdefault(item,0)
	   simSums[item]+=sim
	# Create the normalized list
	 rankings=[(total/simSums[item],item) for item,total in totals.items( )]
	# Return the sorted list
	 rankings.sort()
	 rankings.reverse()
	 return rankings


	#keys = random.sample(dict_test.keys(), 100)
	predictions=[];
	target=[];
	keys = dict_test.keys();
	arr_diff=[];
	tol_count=0
	j = 0;
	for k in keys:
	  j = j+1;
	  print j;
	  r =  getRecommendations(dict_train,k,sim_pearson);
	  for item in r:
	    if item[1] in dict_test[k]:
	      predictions.append(float(item[0]));
	      target.append(dict_test[k][item[1]]);

	print it;
	from sklearn.metrics import mean_squared_error
	mean_error = mean_squared_error(target,predictions)
	file_write.write(str(it)+'\n');
	file_write.write(str(mean_error)+'\n');
