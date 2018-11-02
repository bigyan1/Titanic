import pandas as pd
train_df = pd.read_csv("data/train.csv")
#Dropping columns that are not significant for decision making
train_df = train_df.drop("Name", axis = 1)
train_df = train_df.drop("Fare", axis = 1)
train_df = train_df.drop("Ticket", axis = 1)
#Save PassengerId column to use in the result
passengerId = train_df["PassengerId"]
train_df = train_df.drop("PassengerId", axis = 1)
#Fill Missing values in Age and Embarked columns with mean and mode values
train_df["Age"].fillna(train_df["Age"].mean(), inplace = True)
train_df["Embarked"].fillna(train_df["Embarked"].mode()[0], inplace = True)

