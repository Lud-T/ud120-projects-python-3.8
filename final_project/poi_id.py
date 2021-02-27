#!/usr/bin/python

###Original Import 
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


###My import
from tester import test_classifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pprint
import time

pp = pprint.PrettyPrinter(depth=10)

###First launch bug !
'''First launch
(py2) PS C:\GitHub\\ud120-projects\final_project> python poi_id.py
Traceback (most recent call last):
  File "poi_id.py", line 8, in <module>
    from tester import dump_classifier_and_data
  File "C:\GitHub\\ud120-projects\final_project\tester.py", line 15, in <module>
    from sklearn.cross_validation import StratifiedShuffleSplit
ImportError: No module named cross_validation
Solution :
https://github.com/jkibele/OpticalRS/issues/7
'''

print("  ")
print(" -------")
print("| START |")
print(" -------")
print("  ")


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### First
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)





#########################################
###Data exploration                   ###
#########################################
'''
Begining in python everything takes to me a lot of times to transcode from idea to code.
I began Data exploration with data_dict but I found a way to convert it to a dataframe and it was
easier to me to manipulate the data in it.
It seems there are a lot of code I should have converted to function because now they are working.
When there were not working my goal was to makes them work.
'''
print("  ")
print(" ------------------")
print("| Data Exploration |")
print(" ------------------")
print("  ")
### people number
print(("People number : ", len(data_dict)))
print("  ")
### feature number
print(("features : ", len(list(data_dict.values())[0])))
print("  ")
### feature name
print("features names: ")
### Take the first name I get with 
pp.pprint(list(data_dict['METTS MARK'].keys()))


print("  ")
print("POIs list : ")
print("----")
count = 0
for employee in data_dict:
    if data_dict[employee]['poi'] == True:
        count += 1
        print(employee)
print("----")
print(("Total POIs : ", count))
print("----------")
print("  ")
print("  ")

### rows number
print("----------")
print("  ")
print(("Numbers of rows : ", len(data_dict)))
print("  ")
print("  ")
print(('Ratio POI/nonPOI : ', count/len(data_dict)*100))
print("  ")
print("----------")
print("  ")

print("  ")
print("----------")
print("  ")
print("  ")
print("Person example")
print("  ")
pp.pprint(data_dict['METTS MARK'])
print("  ")
print("----------")
print("  ")

###NaN check
print("Value/Nan check : ")
print("---------------")
### create a dict with all the features to count the NaN
keysNaN = dict((key, 0) for key, value in list(data_dict['METTS MARK'].items()))
###loop in the dict 
for employee in data_dict:
        ###Loop in the dict in the dict
        for key in data_dict[employee]:
            if data_dict[employee][key] == 'NaN':
                #Convert 'NaN' into np.nan to make use of dataframe describe.
                data_dict[employee][key] = np.nan
                #print type(data_dict[employee][key])
                keysNaN[key]+=1
#print "NaN number for each feature:"          
#pp.pprint(keysNaN)
print("---------------")
print("NaN conversion : done ")
print("---------------")



###Zero check
print("---------------")
print("Value/Zero check : ")
print("---------------")
### create a dict with all the features to count the NaN
keysZero = dict((key, 0) for key, value in list(data_dict['METTS MARK'].items()))
for employee in data_dict:
        for key in data_dict[employee]:
            if data_dict[employee][key] == 0:
                keysZero[key]+=1
print("Zero number for each feature:")  
for line in keysZero:
    if keysZero[line] >> 0:
        print((line, " : ", keysZero[line]))
print("---------------")
print("Zero check : done ")
print("---------------")


#########################################
###Data exploration END               ###
#########################################

'''
I check if the NaN count and the zero count can interfere If I replace the NaN by zero !!!
I think For every features without 0 I can put zero instead of NaN

Zero count
 'from_poi_to_this_person': 12,
 'from_this_person_to_poi': 20,

NaN count
 'from_poi_to_this_person': 60,
 'from_this_person_to_poi': 60,

'''

#########################################
###Outliers management                ###
#########################################

print("-------------------")
print("Outliers management : ")
print("-------------------")

###convert dict of dict to Dataframe !
###Take care of type of data
###I used df for plotting and ouliers management
data_df = pd.DataFrame.from_dict(data_dict, orient='index')
#print data_df.describe()
###
###At first, describe was not displaying the whole result I was expecting 
###I used my previous code to convert the NaN to zero, then it worked
###finally I tested the type of the value I was thinking it was NaN but it was string...I convert it to np.nan now it is float
###Now describe is working

'''At this stage I think I am already detecting outliers, one by one
I have done and understood the mini project on enron database which I suppose is relevant
but I feel it is too easy as we know the data well
What about we know nothing about the data and their relations
I tried the IQR method 
I decided in the to display a boxplot and the max value ID for each
'''
###Boxplot method to identify which feature may have outliers
###https://towardsdatascience.com/5-ways-to-detect-outliers-that-every-data-scientist-should-know-python-code-70a54335a623
###https://towardsdatascience.com/detecting-and-treating-outliers-in-python-part-1-4ece5098b755
###
###Manual method IQR
###https://naysan.ca/2020/06/28/interquartile-range-iqr-to-detect-outliers/
###Visualize a boxplot for each column except email adress which is not a number to plot

###first old code plotting every variable.
###I needed to have the name of the outliers, I update code to show each time the maximum value and the ID.
'''
for column in data_df.columns:
    if (column != 'email_address'):
        sns.boxplot(data=data_df[column], orient="h")
        plt.title("Boxplot : "+column)
        plt.show()  
        
        
###Outliers everywhere !!!
###POI is not relevant
###Salary is
###Let's have a closer look to salary
#sns.boxplot(data_df["salary"])
#plt.show()
'''


###Display boxplot and highest value and related name                ###
print("-------------------")
print("Check for the max value : ")
print("-------------------")
###More relevant code
###Very personnal approach to detect outliers
print("-------------------")
print("Display boxplot with the person having the max value for each variable : ")
print("Then, I check in the PDF if it is a mistake or not")
print("We have not many data so I'll try to keep as much data as it makes sense")
print("Also, if we found an outlier presumption from a POI person, it should mean it is not an real outlier")
print("but a real kind of POI detection")
print("-------------------")
outlierMax = []
for column in data_df.columns:
    if (column != 'email_address'):
        if (column != 'poi'):
            #sns.boxplot(data=data_df[column], orient="h")
            search = data_df[column] == data_df[column].max()
            #plt.title(column + "  Max : " + str(data_df[column].max()) + "\nID : " + str(data_df[search].index.values) + "  POI : " + str(data_df.loc[data_df[search].index.values, 'poi'].values))
            ###print column,"\t", data_df[search].index.values,"\t", data_df[column].max(),"\t",data_df.loc[data_df[search].index.values, 'poi'].values
            outlierMax.append([column,str(data_df[search].index.values), float(data_df[column].max()),str(data_df.loc[data_df[search].index.values, 'poi'].values)])
            #plt.show()  

outlierMax_df = pd.DataFrame(outlierMax,columns =['Variable','Name','Value Max','POI'])
pp.pprint(outlierMax_df)

print("   ")
print("As we can see, 'TOTAL' is clearly an outlier.")
print("It is the sum of column values")
print("   ")
print("Higher value are more relevant in this case than lower value, I will not check for them")
print("   ")
print("Checking the PDF provided, I saw the line 'THE TRAVEL AGENCY IN THE PARK' just above")
print("As there is almost no data, this is not a real people and it is not a POI, I decided to drop it either")
print("   ")
print("Negative value : ")
print("Negative are bothering with my approach, let's convert them to absolute.")
print("   ")


###Negative number management ###
print("")
print("---------------")
print("Negatives values check: ")
print("---------------")
print("")
### create a dict with all the features to count the NaN
keysNeg = dict((key, 0) for key, value in list(data_dict['METTS MARK'].items()))
for employee in data_dict:
        for key in data_dict[employee]:
            if type(data_dict[employee][key]) is not str:
                if data_dict[employee][key] < 0:
                    #print employee
                    keysNeg[key]+=1
print("Negative number for each feature:")          
for line in keysNeg:
    if keysNeg[line] >> 0:
        print((line, " : ", keysNeg[line]))

###In the PDF there is no negative value
###I'll convert to positive one
for employee in data_dict:
        for key in data_dict[employee]:
            if type(data_dict[employee][key]) is not str:
                if data_dict[employee][key] < 0:
                    data_dict[employee][key] = abs(data_dict[employee][key])  

print("")
print("---------------")
print("Negatives values check after absolute value conversion: ")
print("---------------")
print("")
### create a dict with all the features to count the NaN
keysNeg = dict((key, 0) for key, value in list(data_dict['METTS MARK'].items()))
for employee in data_dict:
        for key in data_dict[employee]:
            if type(data_dict[employee][key]) is not str:
                if data_dict[employee][key] < 0:
                    #print employee
                    keysNeg[key]+=1
print("Negative number for each feature:")          
for line in keysNeg:
    if keysNeg[line] >> 0:
        print((line, " : ", keysNeg[line]))

data_df = pd.DataFrame.from_dict(data_dict, orient='index')

print("")
print("------------------------------------------------------")
print("Negatives values management DONE ")
print("------------------------------------------------------")
print("")


###enron exercise
'''Before removing outliers

features_list = ["poi","salary", "bonus"]
dataToDisplay = featureFormat(data_dict, features_list, sort_keys = True)
poi=dataToDisplay[:,0]
salary = dataToDisplay[:,1]
bonus = dataToDisplay[:,2]

plt.scatter(salary[poi==1],bonus[poi==1],c='red',s=50,label='poi')
plt.scatter(salary[poi==0],bonus[poi==0],c='blue',s=50,label='not poi')

plt.xlabel("Salary")
plt.ylabel("Bonus")
plt.legend(loc='lower right')
plt.title("Bonus vs Salary")
plt.show()
'''


###IQR testing on two variables
###I did not feel this approach fine with this small dataset.
###I understand IQR testing should be used to automatically remove outliers in large dataset.
###I ended using boxplot and check for each variable who is the most outlying.
###IQR testing
print("")
print("---------------")
print("IQR salary ")
print("---------------")
print("")
IQR = data_df['salary'].quantile(q=0.75) - data_df['salary'].quantile(q=0.25)
q25 = data_df['salary'].quantile(q=0.25)
q75 = data_df['salary'].quantile(q=0.75)

lowerBound = q25 + 1.5*IQR
upperBound = q75 + 1.5*IQR

outliersSalary = data_df[(data_df['salary']>(lowerBound) ) | (data_df['salary']<upperBound )]
print((outliersSalary.sort_values(by=['salary'],ascending=False).head(10)))
print((outliersSalary.shape))


print("")
print("---------------")

print("")
print("---------------")
print("IQR to_messages ")
print("---------------")
print("")
IQR = data_df['to_messages'].quantile(q=0.75) - data_df['to_messages'].quantile(q=0.25)
q25 = data_df['to_messages'].quantile(q=0.25)
q75 = data_df['to_messages'].quantile(q=0.75)

lowerBound = q25 + 1.5*IQR
upperBound = q75 + 1.5*IQR

outliersToMessages = data_df[(data_df['to_messages']>(lowerBound) ) | (data_df['to_messages']<upperBound )]
print((outliersToMessages.sort_values(by=['to_messages'],ascending=False).head(10)))
print((outliersToMessages.shape))

print("")
print("---------------")



print("-----------")
print("Drop outliers : ")
print("-----------")

data_df = data_df.drop(['TOTAL'])
data_dict.pop('TOTAL', 0)
print("Drop 'TOTAL' : done")

data_df = data_df.drop(['THE TRAVEL AGENCY IN THE PARK'])
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
print("Drop 'THE TRAVEL AGENCY IN THE PARK' : done")

print("-------------------")
print("Check for the max value after drops : ")
print("-------------------")



outlierMax = []
for column in data_df.columns:
    if (column != 'email_address'):
        if (column != 'poi'):
            #sns.boxplot(data=data_df[column], orient="h")
            search = data_df[column] == data_df[column].max()
            #plt.title(column + "  Max : " + str(data_df[column].max()) + "\nID : " + str(data_df[search].index.values) + "  POI : " + str(data_df.loc[data_df[search].index.values, 'poi'].values))
            outlierMax.append([column,str(data_df[search].index.values), float(data_df[column].max()),str(data_df.loc[data_df[search].index.values, 'poi'].values)])
            #plt.show()  

outlierMax_df = pd.DataFrame(outlierMax,columns =['Variable','Name','Value Max','POI'])
pp.pprint(outlierMax_df)



print("   ")
print("I check everyone in the PDF")
print("As the max values are correctly reported")
print("I consider the ones below are not outliers")
print("   ")
print("   ")

##########################################################
### Testing purpose ONLY                               ###
##########################################################
###Visualize a boxplot for each column except email adress which is not a number to plot
'''
for column in data_df.columns:
    if (column != 'email_address'):
        sns.boxplot(data=data_df[column], orient="h")
        plt.title("Boxplot : "+column)
        plt.show()  

# sns.boxplot(data=data_df['salary'], orient="h")
# plt.title("Boxplot : "+"Salary"+" - Total")
# plt.show()
'''


###enron exercise
'''###after removing outliers

features_list = ["poi","salary", "bonus"]
dataToDisplay = featureFormat(data_dict, features_list, sort_keys = True)
poi=dataToDisplay[:,0]
salary = dataToDisplay[:,1]
bonus = dataToDisplay[:,2]

plt.scatter(salary[poi==1],bonus[poi==1],c='red',s=50,label='poi')
plt.scatter(salary[poi==0],bonus[poi==0],c='blue',s=50,label='not poi')

plt.xlabel("Salary")
plt.ylabel("Bonus")
plt.legend(loc='lower right')
plt.title("Bonus vs Salary")
plt.show()
'''

'''not relevant
print ""
print "---------------"
print "Employee without payment nor stock: "
print "---------------"
print ""
###New chapter

noPayNoStockPeople=[]
for employee in data_dict:
        if  ((np.isnan(data_dict[employee]['total_payments'])) & (np.isnan(data_dict[employee]['total_stock_value']))) :
            print employee
            noPayNoStockPeople.append(employee)         
print "These person didn't have money data"
print "---------------"





print ""
print "---------------"
print "Employee without relation with POI: "
print "---------------"
print ""
###New chapter

noMailToFromPOI=[]
print "POI \t\t\t Name"
print "----------------------------------------"
for employee in data_dict:
        if  ((np.isnan(data_dict[employee]['from_poi_to_this_person'])) & (np.isnan(data_dict[employee]['from_this_person_to_poi']))) :
            print data_dict[employee]['poi'], "\t\t", employee
            noMailToFromPOI.append(employee)         
print "----"


print ""
print "---------------"
print "Employee without relation with POI AND are not POI : "
print "---------------"
print ""
###New chapter

noMailToFromPOI=[]
print "POI \t\t\t Name"
print "----------------------------------------"
for employee in data_dict:
        if  ((np.isnan(data_dict[employee]['from_poi_to_this_person'])) & (np.isnan(data_dict[employee]['from_this_person_to_poi'])) & (data_dict[employee]['poi']==False)) :
            print data_dict[employee]['poi'], "\t\t", employee
            noMailToFromPOI.append(employee)         
print "----"
'''
###I have identified these people
###When I'll try to classify, I'll check the accuracy with and without them.
###At this stage I think the mail relation are pretty important
###So to me, these guys will work as examples to raise accuracy.
###Let's check if I understood how it works or not !


''' Test code for me
for employee in data_dict:
        for name in noMailToFromPOI:
            if name == employee:
                print name

print "---------------"
'''

###I'll identify the highest
'''
print "  "
print " ------------------"
print "| Salary top 3  |"
print " ------------------"
print "  "
print data_df.sort_values(by=['salary'],ascending=False).head(3)

print "  "
print " ------------------"
print "| Salary bottom 3  |"
print " ------------------"
print "  "
print data_df.sort_values(by=['salary'],ascending=True).head(3)
'''

'''
search = data_df["salary"] == data_df["salary"].max()
print data_df[search].index.values
'''

###destroy the line TOTAL I found then check again for outliers
#data_dict.pop(data_df[search].index.values[0])
###check if it worked
#print "People number : ", len(data_dict)
#data_df = data_df.drop(data_df[search].index)

###enron exercise
###After removing outliers
'''
features_list = ["poi","salary", "bonus"]
dataToDisplay = featureFormat(data_dict, features_list, sort_keys = True)
poi=dataToDisplay[:,0]
salary = dataToDisplay[:,1]
bonus = dataToDisplay[:,2]

plt.scatter(salary[poi==1],bonus[poi==1],c='red',s=50,label='poi')
plt.scatter(salary[poi==0],bonus[poi==0],c='blue',s=50,label='not poi')

plt.xlabel("Salary")
plt.ylabel("Bonus")
plt.legend(loc='lower right')
plt.title("Bonus vs Salary")
plt.show()
'''

#pp.pprint(data_df.describe())

'''
sns.boxplot(data_df["salary"])
plt.show()
'''

'''not relevant in the end, just testing
print ""
print "---------------"
print "quantiles another way: "
print "---------------"
print ""
###New chapter
###https://stackoverflow.com/questions/33518472/how-to-get-boxplot-data-for-matplotlib-boxplots
###https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51
quantiles = data_df.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
pp.pprint(quantiles)



print ""
print "---------------"
print "Detecting outliers by sorting : SALARY descending"
print "---------------"
print ""

pp.pprint(data_df.sort_values(by='salary', ascending=False).head(10))

print ""
print "---------------"
print "Detecting outliers by sorting : SALARY ascending"
print "---------------"
print ""

pp.pprint(data_df.sort_values(by='salary', ascending=True, na_position='last').head(10))
'''

##########################################################
### Testing purpose only END                           ###
##########################################################




print("")
print("---------------")
print("compare with PDF: ")
print("---------------")
print("")


print("'deferred_income' data has been converted in negative value")
print("Whereas boxplot may notify about outliers, it seems it is mostly dispersion of the data")
print("Outliers is not about detecting who's guilty or not, just if the data is relevant or not !")
print("I have to keep as much as I can data")
print("To me, the only real outlier is TOTAL and The travel agency park")
print("")






#####################################################################################################

###Features selection


print("")
print("------------------")
print("Features selection : ")
print("------------------")
print("")


print("")
print("At first glance, the first features I'll like to add :")
print("Ratio of mail from and to POI per person")
print("And the ratio POI related from the total mail")
print("I imagine it is the preferred channel of communiation between people")
print("So I try to get a max of data of this interaction")
print("")



###new features

for line in data_dict:
    #Ratio sent mail total and sent mail to POI
    data_dict[line]['ratio_to_poi'] = float(data_dict[line]['from_this_person_to_poi']) / float(data_dict[line]['from_messages'])
    #Ratio received mail total and received mail to POI
    data_dict[line]['ratio_from_poi'] = float(data_dict[line]['from_poi_to_this_person']) / float(data_dict[line]['to_messages'])

for line in data_dict:
    #Total mail : sent + received
    data_dict[line]['total_mail'] = data_dict[line]['from_messages'] + data_dict[line]['to_messages']
    #total mail related to POI
    data_dict[line]['total_mail_poi'] = data_dict[line]['from_poi_to_this_person'] + data_dict[line]['from_this_person_to_poi']
    #Ratio total mail and total mail related to POI
    data_dict[line]['total_ratio_mail_poi'] = float(data_dict[line]['total_mail_poi']) / float(data_dict[line]['total_mail'])


###Others features calculation do not seem to be as relevant as thes ones.

'''
features_list = ['poi','total_mail','total_mail_poi']
dataToDisplay = featureFormat(data_dict, features_list, sort_keys = True)
poi=dataToDisplay[:,0]
a = dataToDisplay[:,1]
b = dataToDisplay[:,2]

plt.scatter(a[poi==1],b[poi==1],c='red',label='poi')
plt.scatter(a[poi==0],b[poi==0],c='blue',label='not poi')

plt.xlabel(features_list[1])
plt.ylabel(features_list[2])
plt.legend(loc='upper right')
plt.title(features_list[1] + " vs " +  features_list[2])
plt.show()
'''

###Ratio money
'''
features_list = ['poi','salary','bonus']
dataToDisplay = featureFormat(data_dict, features_list, sort_keys = True)
poi=dataToDisplay[:,0]
a = dataToDisplay[:,1]
b = dataToDisplay[:,2]

plt.scatter(a[poi==1],b[poi==1],c='red',label='poi')
plt.scatter(a[poi==0],b[poi==0],c='blue',label='not poi')

plt.xlabel(features_list[1])
plt.ylabel(features_list[2])
plt.legend(loc='upper right')
plt.title(features_list[1] + " vs " +  features_list[2])
plt.show()
'''



###Ratio message relative to POI
'''
features_list = ['poi','from_poi_to_this_person','to_messages','from_this_person_to_poi','from_messages']
dataToDisplay = featureFormat(data_dict, features_list, sort_keys = True)
poi=dataToDisplay[:,0]
a = dataToDisplay[:,1]/dataToDisplay[:,2]
b = dataToDisplay[:,3]/dataToDisplay[:,4]

plt.scatter(a[poi==1],b[poi==1],c='red',label='poi')
plt.scatter(a[poi==0],b[poi==0],c='blue',label='not poi')

plt.xlabel(features_list[1] + "/" + features_list[2])
plt.ylabel(features_list[3] + "/" + features_list[4])
plt.legend(loc='upper right')
plt.title(features_list[1] + "/" + features_list[2] + " vs " + features_list[3] + "/" + features_list[4])
plt.show()
'''

###Ratio stock
'''
features_list = ['poi','exercised_stock_options','total_stock_value']
dataToDisplay = featureFormat(data_dict, features_list, sort_keys = True)
poi=dataToDisplay[:,0]
a = dataToDisplay[:,1]
b = dataToDisplay[:,2]

plt.scatter(a[poi==1],b[poi==1],c='red',label='poi')
plt.scatter(a[poi==0],b[poi==0],c='blue',label='not poi')

plt.xlabel(features_list[1])
plt.ylabel(features_list[2])
plt.legend(loc='upper right')
plt.title(features_list[1] + " vs " +  features_list[2])
plt.show()
'''



'''
###Total payment and stock
features_list = ['poi','total_payments','total_stock_value']
dataToDisplay = featureFormat(data_dict, features_list, sort_keys = True)
poi=dataToDisplay[:,0]
a = dataToDisplay[:,1]
b = dataToDisplay[:,2]

plt.scatter(a[poi==1],b[poi==1],c='red',label='poi')
plt.scatter(a[poi==0],b[poi==0],c='blue',label='not poi')

plt.xlabel(features_list[1])
plt.ylabel(features_list[2])
plt.legend(loc='upper right')
plt.title(features_list[1] + " vs " +  features_list[2])
plt.show()
### I feel no correlation there
'''


'''###bonus and exercised stock
features_list = ['poi','deferred_income','restricted_stock_deferred']
dataToDisplay = featureFormat(data_dict, features_list, sort_keys = True)
poi=dataToDisplay[:,0]
a = dataToDisplay[:,1]
b = dataToDisplay[:,2]

plt.scatter(a[poi==0],b[poi==0],c='blue',label='not poi')
plt.scatter(a[poi==1],b[poi==1],c='red',label='poi')


plt.xlabel(features_list[1])
plt.ylabel(features_list[2])
plt.legend(loc='upper right')
plt.title(features_list[1] + " vs " +  features_list[2])
plt.show()
### I feel no correlation there
### I swaped order of POI to highlight POI in red in the graph.
### Let's put 0 instead of NaN and check it again, intuitively, I cannot understand there is no correlation in this test.
'''


###Finally I need to convert NaN to zero to explore relationship between features
###NaN to zero
print("Value/Nan check : ")
print("---------------")
###https://stackoverflow.com/questions/36000993/numpy-isnan-fails-on-an-array-of-floats-from-pandas-dataframe-apply/36001292
###isnan and float !!!
#keysNaN = dict((key, 0) for key, value in data_dict['METTS MARK'].iteritems())
for employee in data_dict:
        for key in data_dict[employee]:
            if pd.isnull(data_dict[employee][key]):
                data_dict[employee][key]=0


'''###bonus and exercised stock
features_list = ['poi','deferred_income','restricted_stock_deferred']
dataToDisplay = featureFormat(data_dict, features_list, sort_keys = True)
poi=dataToDisplay[:,0]
a = dataToDisplay[:,1]
b = dataToDisplay[:,2]

plt.scatter(a[poi==0],b[poi==0],c='blue',label='not poi')
plt.scatter(a[poi==1],b[poi==1],c='red',label='poi')


plt.xlabel(features_list[1])
plt.ylabel(features_list[2])
plt.legend(loc='upper right')
plt.title(features_list[1] + " vs " +  features_list[2])
plt.show()
'''### still no correlation, let's test an selectKbest


'''
features_list = ["poi","ratio_to_poi", "ratio_from_poi"]
dataToDisplay = featureFormat(data_dict, features_list, sort_keys = True)
poi=dataToDisplay[:,0]
a = dataToDisplay[:,1]
b = dataToDisplay[:,2]

plt.scatter(a[poi==1],b[poi==1],c='red',label='poi')
plt.scatter(a[poi==0],b[poi==0],c='blue',label='not poi')

plt.xlabel(features_list[1])
plt.ylabel(features_list[2])
plt.legend(loc='upper right')
plt.title(features_list[1] + " vs " +  features_list[2])
plt.show()

### It seems there is a corelation let's test antoher
'''



features_list = ['poi','to_messages','deferral_payments','expenses','deferred_income','long_term_incentive','restricted_stock_deferred','shared_receipt_with_poi','loan_advances','from_messages','other','director_fees','bonus','total_stock_value','from_poi_to_this_person','ratio_to_poi','from_this_person_to_poi','restricted_stock','salary','total_payments','total_mail','exercised_stock_options','total_mail_poi','ratio_from_poi','total_ratio_mail_poi']

###ff is the list features_list without poi
ff = ['to_messages','deferral_payments','expenses','deferred_income','long_term_incentive','restricted_stock_deferred','shared_receipt_with_poi','loan_advances','from_messages','other','director_fees','bonus','total_stock_value','from_poi_to_this_person','ratio_to_poi','from_this_person_to_poi','restricted_stock','salary','total_payments','total_mail','exercised_stock_options','total_mail_poi','ratio_from_poi','total_ratio_mail_poi']

print("")
print("-------------------------")
print("Final Features selection : ")
print("-------------------------")
print("")


pp.pprint(features_list)
pp.pprint(len(features_list))

print("-------------------------")
print("removed : ")
print("\t'email_address' => ID information")
print("added : ")
print("\t'ratio_to_poi'\t\t => Ratio sent mail total and sent mail to POI")
print("\t'ratio_from_poi'\t => received mail total and received mail to POI")
print("\t'total_mail'\t\t => Total mail : sent + received")
print("\t'total_mail_poi'\t => total mail related to POI : sent + received to/from POI")
print("\t'total_ratio_mail_poi'\t => Ratio total mail and total mail related to POI")
print("-------------------------")
print("")




###features_list = ["poi", "salary", "bonus"] 
###data_array = featureFormat( data_dictionary, features_list )
###label, features = targetFeatureSplit(data_array)
### these two lines extract the features specified in features_list
### and extract them from data_dict, returning a numpy array
data_array = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data_array)
print("length of features after processing")

print("")
print("SelectKBest method to highlight the best features to use")
print("")


###SelectKbest
###output nicely
###https://stackoverflow.com/questions/41897020/sklearn-selectkbest-how-to-create-a-dict-of-feature1score-feature2score
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif)
selector.fit(features, labels)
combined = list(zip(ff, selector.scores_))
combined.sort(reverse=True, key= lambda x: x[1])
kbest_df = pd.DataFrame(combined)
pp.pprint(kbest_df)

'''###BAD SelectKbest results !?? Yes columns names included POI and shifted everything...
###
###
#                            0          1
# 0                  total_mail  25.097542
# 1                       bonus  24.464726
# 2               director_fees  21.060002
# 3            restricted_stock  18.575703
# 4     from_poi_to_this_person  16.641707
# 5                    expenses  11.595548
# 6             deferred_income  10.072455
# 7     from_this_person_to_poi   8.961784
# 8                      salary   8.866722
# 9   restricted_stock_deferred   8.746486
# 10    shared_receipt_with_poi   7.242730
# 11          deferral_payments   6.234201
# 12             ratio_from_poi   5.518506
# 13          total_stock_value   5.344942
# 14    exercised_stock_options   4.955198
# 15              from_messages   4.204971
# 16             total_mail_poi   3.210762
# 17               ratio_to_poi   2.426508
# 18                      other   2.107656
# 19                        poi   1.698824
# 20             total_payments   0.515192
# 21        long_term_incentive   0.245090
# 22                to_messages   0.225355
# 23              loan_advances   0.164164
'''

###GOOD results !!!!!
###My table has shifted because I let 'POI' in the features list to display !
# 0     exercised_stock_options  24.815080
# 1           total_stock_value  24.179972
# 2                       bonus  20.792252
# 3                      salary  18.289684
# 4                ratio_to_poi  16.409713
# 5             deferred_income  11.458477
# 6         long_term_incentive   9.922186
# 7            restricted_stock   8.828679
# 8              total_payments   8.772778
# 9     shared_receipt_with_poi   8.589421
# 10              loan_advances   7.184056
# 11                   expenses   6.094173
# 12       total_ratio_mail_poi   5.399370
# 13    from_poi_to_this_person   5.243450
# 14             total_mail_poi   4.863682
# 15                      other   4.187478
# 16             ratio_from_poi   3.128092
# 17    from_this_person_to_poi   2.382612
# 18              director_fees   2.126328
# 19                to_messages   1.646341
# 20                 total_mail   0.490666
# 21  restricted_stock_deferred   0.247053
# 22          deferral_payments   0.233091
# 23              from_messages   0.169701



###
###My final list of features will be :
###

print("")
print("My features list will be :")
print("")

###Bad features selection !
#features_list =  ['poi', 'total_mail', 'bonus', 'director_fees', 'restricted_stock', 'from_poi_to_this_person', 'expenses', 'deferred_income', 'from_this_person_to_poi', 'salary', 'restricted_stock_deferred']

###Good feature selection !!!!!!!!!!!
features_list = ['poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'ratio_to_poi', 'deferred_income', 'long_term_incentive','restricted_stock', 'total_payments', 'shared_receipt_with_poi', 'loan_advances', 'expenses', 'total_ratio_mail_poi', 'from_poi_to_this_person']


pp.pprint(features_list)

print("")
print("end of features list")
print("")


#####################################################################################################
###First
###https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
###This will help me to determine which classifier I'll choose to test
###According to the diagram
###>50
###Predicting a category Yes
###Labeled Data Yes
###<100K Yes
###LinearSVC
###Text data NO
###KNeighborsClassifier
###if not working : It is not working whithout tuning
###adaboost
###
###GaussianNB


### Task 2: Remove outliers DONE
### Task 3: Create new feature(s) DONE
### Store to my_dataset for easy export below. 
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

##############################################################################################
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

#from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier(n_neighbors=3, weights='uniform',leaf_size=5,algorithm='auto', p=1)


# from sklearn.svm import LinearSVC
# clf = LinearSVC()

#from sklearn.tree import DecisionTreeClassifier
#clf = DecisionTreeClassifier(max_features='auto', splitter='best', criterion='entropy')

# from sklearn.ensemble import AdaBoostClassifier
# clf = AdaBoostClassifier()


start = time.time()
print("test_classifier")
print("3 2 1... go !")
print("")

test_classifier(clf,my_dataset,features_list)

end = time.time()
print("")
print((end - start, " secs"))
print("")
print("...finish")
print("")

##############################################################################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html





'''GridSearchCV
start = time.time()
print ""
print "GridSearchCV"
print "3 2 1... go !"
print ""


from sklearn.model_selection import GridSearchCV
classifier = KNeighborsClassifier()

#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
#As F1 is F1 = 2 * (precision * recall) / (precision + recall)
#I think it is the best one to benchmark we are looking for precision and recall at least 0.3
score_metric = 'f1'

params = {'n_neighbors': range(1, 11),
          'weights': ['uniform', 'distance'],
          'algorithm': ['auto', 'ball_tree','kd_tree','brute'],
          'leaf_size': range(5, 30),
          'p': [1, 2]
          }

search = GridSearchCV(estimator=classifier, param_grid=params, scoring=score_metric, n_jobs=1, refit=True, cv=10)
search.fit(features, labels)

print "Best parameters: ", search.best_params_
print "Best Score: ", score_metric, search.best_score_

end = time.time()
print ""
print end - start, " secs"
print ""
print "...finish"
'''



# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#####################################################################################################
###WARNING !
###After a few manual tries, I was surprised about how bad were the results.
###My first step was to edit the features list, keeping only what I feel and not what I calculate with selectKBest
###It was so different I decided to check again the code !
###WARNING !



dump_classifier_and_data(clf, my_dataset, features_list)



print("  ")
print(" -----")
print("| END |")
print(" -----")
print("  ")




#Final results !!!

# test_classifier
# 3 2 1... go !

# DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
            # max_features='auto', max_leaf_nodes=None,
            # min_impurity_decrease=0.0, min_impurity_split=None,
            # min_samples_leaf=1, min_samples_split=2,
            # min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            # splitter='best')
        # Accuracy: 0.83553       Precision: 0.37107      Recall: 0.33600 F1: 0.35266     F2: 0.34247
        # Total predictions: 15000        True positives:  672    False positives: 1139   False negatives: 1328   True negatives: 11861


# 0.561000108719  secs

# ...finish


# GridSearchCV
# 3 2 1... go !

# C:\Anaconda3\envs\py2\lib\site-packages\sklearn\metrics\classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
  # 'precision', 'predicted', average, warn_for)
# Best parameters:  {'max_features': 'sqrt', 'splitter': 'best', 'criterion': 'gini'}
# Best Score:  f1 0.4638694638694638

# 0.220999956131  secs

# ...finish


###final final result
# test_classifier
# 3 2 1... go !

# KNeighborsClassifier(algorithm='auto', leaf_size=1, metric='minkowski',
           # metric_params=None, n_jobs=1, n_neighbors=3, p=1,
           # weights='uniform')
        # Accuracy: 0.87880       Precision: 0.60156      Recall: 0.26950 F1: 0.37224     F2: 0.30295
        # Total predictions: 15000        True positives:  539    False positives:  357   False negatives: 1461   True negatives: 12643


# 1.12999987602  secs

# ...finish


# GridSearchCV
# 3 2 1... go !

# C:\Anaconda3\envs\py2\lib\site-packages\sklearn\metrics\classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
  # 'precision', 'predicted', average, warn_for)
# Best parameters:  {'n_neighbors': 5, 'weights': 'uniform', 'leaf_size': 15, 'algorithm': 'auto'}
# Best Score:  f1 0.18764568764568765

# 4.25  secs

# ...finish


###first results with bad features selection#########################################################

# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',metric_params=None, n_jobs=1, n_neighbors=5, p=2,weights='uniform')
# Accuracy: 0.82657       Precision: 0.04852      Recall: 0.01150 F1: 0.01859     F2: 0.01357
# Total predictions: 14000        True positives:   23    False positives:  451   False negatives: 1977   True negatives: 11549

# GaussianNB(priors=None)
# Accuracy: 0.34543       Precision: 0.17341      Recall: 0.95100 F1: 0.29334     F2: 0.50137
# Total predictions: 14000        True positives: 1902    False positives: 9066   False negatives:   98   True negatives: 2934




###first results with good features selection !########################################################

# GaussianNB(priors=None)
# Accuracy: 0.82193       Precision: 0.32480      Recall: 0.31100 F1: 0.31775     F2: 0.31367
# Total predictions: 15000        True positives:  622    False positives: 1293   False negatives: 1378   True negatives: 11707

# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=5, p=2, weights='uniform')
# Accuracy: 0.87640       Precision: 0.63878      Recall: 0.16800 F1: 0.26603     F2: 0.19704
# Total predictions: 15000        True positives:  336    False positives:  190   False negatives: 1664   True negatives: 12810

# LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
# intercept_scaling=1, loss='squared_hinge', max_iter=1000,
# multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
# verbose=0)
# Accuracy: 0.68300       Precision: 0.13335      Recall: 0.25050 F1: 0.17405     F2: 0.21306
# Total predictions: 15000        True positives:  501    False positives: 3256   False negatives: 1499   True negatives: 9744

# LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
# intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',
# penalty='l2', random_state=None, tol=0.0001, verbose=0)
# Accuracy: 0.69147       Precision: 0.13090      Recall: 0.23300 F1: 0.16763     F2: 0.20156
# Total predictions: 15000        True positives:  466    False positives: 3094   False negatives: 1534   True negatives: 9906

# LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,intercept_scaling=1, loss='squared_hinge', max_iter=1000, multi_class='ovr', penalty='l2', random_state=None, tol=0.0001, verbose=0)
# Accuracy: 0.69073       Precision: 0.13070      Recall: 0.23350 F1: 0.16759     F2: 0.20176
# Total predictions: 15000        True positives:  467    False positives: 3106   False negatives: 1533   True negatives: 9894

# DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False, random_state=None, splitter='best')
# Accuracy: 0.81380       Precision: 0.29509      Recall: 0.28550 F1: 0.29022     F2: 0.28737
# Total predictions: 15000        True positives:  571    False positives: 1364   False negatives: 1429   True negatives: 11636
# 0.700000047684  secs

# AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0, n_estimators=50, random_state=None)
# Accuracy: 0.83247       Precision: 0.34293      Recall: 0.28000 F1: 0.30829     F2: 0.29067
# Total predictions: 15000        True positives:  560    False positives: 1073   False negatives: 1440   True negatives: 11927
# 67.7769999504  secs




# DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False, random_state=None, splitter='best')
# Accuracy: 0.82960       Precision: 0.34417      Recall: 0.30700 F1: 0.32452     F2: 0.31378
# Total predictions: 15000        True positives:  614    False positives: 1170   False negatives: 1386   True negatives: 11830
# 0.8109998703  secs




###after grid auto !!
# test_classifier
# 3 2 1... go !

# DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
            # max_features='auto', max_leaf_nodes=None,
            # min_impurity_decrease=0.0, min_impurity_split=None,
            # min_samples_leaf=1, min_samples_split=2,
            # min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            # splitter='best')
        # Accuracy: 0.82767       Precision: 0.33705      Recall: 0.30250 F1: 0.31884     F2: 0.30883
        # Total predictions: 15000        True positives:  605    False positives: 1190   False negatives: 1395   True negatives: 11810


# 0.559000015259  secs

# ...finish





###first grid test
# test_classifier
# 3 2 1... go !

# KNeighborsClassifier(algorithm='auto', leaf_size=1, metric='minkowski',
           # metric_params=None, n_jobs=1, n_neighbors=3, p=1,
           # weights='uniform')
        # Accuracy: 0.87880       Precision: 0.60156      Recall: 0.26950 F1: 0.37224     F2: 0.30295
        # Total predictions: 15000        True positives:  539    False positives:  357   False negatives: 1461   True negatives: 12643


# 1.13000011444  secs

# ...finish
# GridSearchCV
# 3 2 1... go !

# C:\Anaconda3\envs\py2\lib\site-packages\sklearn\metrics\classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
  # 'precision', 'predicted', average, warn_for)
# Best parameters:  {'n_neighbors': 3, 'weights': 'uniform', 'leaf_size': 1, 'algorithm': 'auto', 'p': 1}
# Best Score:  f1 0.29160839160839164

# 53.0279998779  secs

# ...finish


# GridSearchCV
# 3 2 1... go !

# C:\Anaconda3\envs\py2\lib\site-packages\sklearn\metrics\classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
  # 'precision', 'predicted', average, warn_for)
# Best parameters:  {'max_features': 'auto', 'splitter': 'best', 'criterion': 'entropy'}
# Best Score:  f1 0.4673659673659673

# 0.228000164032  secs

# ...finish



