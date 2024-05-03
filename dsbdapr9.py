import seaborn as sns
dataset=sns.load_dataset('titanic')
dataset.head()

sns.boxplot(x='sex',y='age',data=dataset)

sns.boxplot(x='sex',y='age',data=dataset,hue='survived')
