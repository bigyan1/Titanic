import pandas as pd
from sklearn.model_selection import train_test_split
#Read train and test csv into dataframe
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
#Dropping columns that are not significant for decision making
train_df = train_df.drop("Name", axis = 1)
train_df = train_df.drop("Fare", axis = 1)
train_df = train_df.drop("Ticket", axis = 1)
#Similarly drop for test dataframe
test_df = test_df.drop(["Name", "Fare", "Ticket"], axis = 1)
#Save PassengerId column to use in the result
passengerId = train_df["PassengerId"]
train_df = train_df.drop("PassengerId", axis = 1)
#Save passengerId for test dataframe
passengerId_test = test_df["PassengerId"]
test_df = test_df.drop("PassengerId", axis = 1)
#Fill Missing values in Age and Embarked columns with mean and mode values
train_df["Age"].fillna(train_df["Age"].mean(), inplace = True)
train_df["Embarked"].fillna(train_df["Embarked"].mode()[0], inplace = True)
#Replace 'male' with 1 and 'female' with 0 in Sex column
train_df['Sex'].replace("male", 1, inplace = True)
train_df['Sex'].replace("female", 1, inplace = True)
#Split the features column and 'Survived' column
train_Y = train_df["Survived"]
train_X = train_df.drop("Survived", axis =1)
#Converting categorical value in 'Embarked' column into a new column
train_X_onehot = pd.get_dummies(train_X, columns = ["Embarked"])
test_onehot = pd.get_dummies(test_df, columns = ["Embarked"])
#Split the train and test data
train_X_train, train_X_test, train_Y_train, train_Y_test = train_test_split(train_X_onehot, train_Y, test_sie = 0.2)


