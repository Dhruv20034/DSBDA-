import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dataset=sns.load_dataset('titanic')
print(dataset)
dataset.head()

import seaborn as sns
sns.distplot(x = dataset['age'], bins = 10)

sns.distplot(dataset['age'], bins = 10,kde=False)

import seaborn as sns
sns.jointplot(x = dataset['age'], y = dataset['fare'], kind ='scatter')

sns.jointplot(x = dataset['age'], y = dataset['fare'], kind = 'hex')

sns.rugplot(dataset['fare'])

sns.barplot(x='sex', y='age', data=dataset)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.barplot(x='sex', y='age', data=dataset, estimator=np.std)

sns.countplot(x='sex', data=dataset)

sns.boxplot(x='sex', y='age', data=dataset)

sns.boxplot(x='sex', y='age', data=dataset, hue="survived")

sns.violinplot(x='sex', y='age', data=dataset)

sns.violinplot(x='sex', y='age', data=dataset, hue='survived')

sns.stripplot(x='sex', y='age', data=dataset, jitter=False)

sns.stripplot(x='sex', y='age', data=dataset, jitter=True)

sns.stripplot(x='sex', y='age', data=dataset, jitter=True, hue='survived')

sns.swarmplot(x='sex', y='age', data=dataset)

sns.swarmplot(x='sex', y='age', data=dataset, hue='survived')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dataset = sns.load_dataset('titanic')
dataset.head()

import seaborn as sns
dataset = sns.load_dataset('titanic') 
sns.histplot(dataset['fare'], kde=False,bins=10)
