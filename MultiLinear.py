import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df = pd.read_csv('manhattan.txt')

x = df[['bedrooms','bathrooms', 'size_sqft', 'min_to_subway', 
'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 
'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

y = df['rent']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2, train_size = .8, random_state = 6)

mlr = LinearRegression()
model = mlr.fit(x_train, y_train)

y_predict = mlr.predict(x_test)


#Predicting rent for a random apartment
apartment = [[1, 1, 620, 16, 1, 98, 1, 0, 1, 0, 0, 1, 1, 0]]
rent_predict = mlr.predict(apartment)
#print('predicted rent: ${}'.format(rent_predict))


#Scatter plot of real rent vs predicted rent
'''
plt.scatter(y_test, y_predict)
plt.title("Real Rent vs Predicted Rent")
plt.xlabel('Real Price')
plt.ylabel('Predicted Price')
plt.show()
'''

#print(model.coef_)

#These aspects have no coorelation, thus it is hard to draw any useful conclusions
'''
plt.scatter(df['size_sqft'], df['rent'], alpha=.4)
plt.scatter(df['min_to_subway'], df['rent'], alpha=.4)
plt.scatter(df['building_age_yrs'], df['no_fee'])
plt.show()
'''

#find r^2 values
test_score = mlr.score(x_test, y_test)
train_score = mlr.score(x_train, y_train)
print('Training score: {}\n'.format(train_score),'Test score: {}'.format(test_score))
#R^2 test score is just good enough to keep this model.  Ideally it would be as close to .9 as possible.
#Will assess model with other NYC boroughs to check accuracy

brooklyndf = pd.read_csv('brooklyn.txt')
queensdf = pd.read_csv('queens.txt')
combineddf = pd.read_csv('streeteasy.txt')

x1 = brooklyndf[['bedrooms','bathrooms', 'size_sqft', 'min_to_subway', 
'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 
'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

y1 = brooklyndf['rent']

x2 = queensdf[['bedrooms','bathrooms', 'size_sqft', 'min_to_subway', 
'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 
'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

y2 = queensdf['rent']

x3 = combineddf[['bedrooms','bathrooms', 'size_sqft', 'min_to_subway', 
'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 
'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

y3 = combineddf['rent']

x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1, test_size=.2, train_size=.8, random_state = 6)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2,y2, test_size =.2, train_size=.8, random_state= 6)
x3_train, x3_test, y3_train, y3_test = train_test_split(x3,y3, test_size =.2, train_size=.8, random_state= 6)

brooklyn_model = mlr.fit(x1_train, y1_train)
brooklyn_rent = mlr.predict(x1_test)

queens_model = mlr.fit(x2_train, y2_train)
queens_rent = mlr.predict(x2_test)

combined_model = mlr.fit(x3_train, y3_train)
combined_rent = mlr.predict(x3_test)

#Plotting model for all NYC boroughs

'''
plt.scatter(y1_test, brooklyn_rent)
plt.title("Rent in Brooklyn vs Predicted Rent")
plt.xlabel('Real Price')
plt.ylabel('Predicted Price')
plt.show()
plt.close()


plt.scatter(y2_test, queens_rent)
plt.title("Rent in Queens vs Predicted Rent")
plt.xlabel('Real Price')
plt.ylabel('Predicted Price')
plt.show()
plt.close()

plt.scatter(y3_test, combined_rent)
plt.title("Rent in NYC vs Predicted Rent")
plt.xlabel('Real Price')
plt.ylabel('Predicted Price')
plt.show()
plt.close()
'''

brooklyn_test_score = mlr.score(x1_test, y1_test)
brooklyn_train_score = mlr.score(x1_train, y1_train)

queens_test_score = mlr.score(x2_test, y2_test)
queens_train_score = mlr.score(x2_train, y2_train)

combined_test_score = mlr.score(x3_test, y3_test)
combined_train_score = mlr.score(x3_train, y3_train)

print('Brooklyn Training score: {}\n'.format(brooklyn_train_score),'Brooklyn Test score: {}'.format(brooklyn_test_score))
print('Queens Training score: {}\n'.format(queens_train_score),'Queens Test score: {}'.format(queens_test_score))
print('Combined Training score: {}\n'.format(combined_train_score),'Combined Test score: {}'.format(combined_test_score))

apartment = [[1, 1, 620, 16, 1, 98, 1, 0, 1, 0, 0, 1, 1, 0]]
rent_predict = mlr.predict(apartment)
print(rent_predict)

#CONCLUSIONS:
'''
Because the places available in Manhattan are so different from the places for rent in Brooklyn or Quens,
it is very likely that there will not be a single model that can accuratley predict prices in all three
areas. This is evidenced by the fact that despite poor R^2 scores in Queens and Brooklyn, the combined
model still scored approximately .7, just past the range of being considered useful.
'''