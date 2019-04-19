# Self organizing map

Alghorithm for self organizing map. It is used to find optimal data representations based on feature vectors.
There are two example data sets that you can use to test the alghorithm.

## Getting Started

### Prerequisites

First, you need to install these libraries:

* pandas
* matplotlib
* sklearn

### Importing library to your python script

Clone the repository or download zip version and copy "SelfOrganizingMap.py" to the folder with your python script.
To run the code you have to import the SOM class and pandas library:

```
import pandas as pd
import SelfOrganizingMap
```

Then import your set of data from .csv file:

```
data = pd.read_csv("irisData.csv", header=0)
```

Initialize SOM class with specified parameters:

```
som = SelfOrganizingMap.SOM(data, no_of_neurons=55, learning_rate=0.5, iterations=100, initial_neighbourhood=0.2,
                      dead_neur_percent=0.06, delete_neurons=False, display_neighbourhood=False,
                      display_current_iteration=True, display_animation=False, skip_frames_count=1)
```

Finally run the alghorithm:

```
som.run()
```

## List of parameters

* data - your set of data to analyze
* no_of_neurons - number of neurons you want to use to analyze the data (default 25)
* learning_rate - starting rate at which neurons learn (default 0.5)
* iterations - number of epochs in which neurons are going to train (default 1000) 
* initial_neighbourhood - starting neighbourhood radius (default 0.5)
* dead_neur_percent - decimal value in range 0.00 - 1.00 - percent of max iterations - used to delete dead neurons every such percent (default 0.14)
* delete_neurons - boolean value - specify whether you want neurons to be deleted every dead_neur_percent percent (default False)
* display_neighbourhood - boolean value - display neighbourhood radius on plot (default False)
* display_current_iteration - boolean value - print current epoch to the console (default False)
* display_animation - boolean value - specify whether you want to watch neurons on plot during learning (default True)
* skip_frames_count - if you want to reduce wait time during learning use it not to draw every epoch on plot (default 1)