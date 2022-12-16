# Goals
## - Acquire data on roller coasters to find drivers of coaster length
## - Use those drivers to build a model to predict coaster length
## - Use that model to attempt to predict lengths of roller coasters


# Plan
- Acquire data from Kaggle
- Prepare data
    - Remove roller coasters which are rumored, under construction, or closed
	- Create engineered features using existing data:
		- man_groups
- Explore data in search of predictors of coaster length
	- Answer the following initial questions
        - Is there a linear relationship between speed and length?
        - Do steel roller coasters vary significantly in length from wooden ones?
        - Is there a relationship between coaster manufacturer and length?
		- Could clustering on `speed` and `man_group` improve model performance?
- Model data

- Draw conclusions

# Data Dictionary
Variable Name | Data Type | Definition
--- | --- | ---
**name** | *string* |The name of the roller coaster
**steel_track** | boolean | Whether or not the track of the roller coaster is primarily steel
**seating_type** | category | The style of roller coaster (e.g. Sit Down, Stand Up, Inverted)
**speed** | *float* | the maximum speed of the roller coaster in km/h.
**height** | *float* | the maximum height of the roller coaster in meters.
**length** | *float* | the length of the roller coaster in meters (NOT in feet as Kaggle states)
**num_inversions** | *int* | The number of inversions on the roller coaster
**manufacturer** | *string* | The manufacturer of the roller coaster
**park** | *string* | The park the roller coaster is located at
**man_group** | *int* | value derrived from which manufacturers produce the longest roller coasters, grouped in groups of 5

# Steps to Reproduce:
1. download data set from Kaggle
2. Run `final_report.ipynb`

# Conclusions
## Exploration
- The most apparent drivers of coaster length are speed and manufacturer (which, for simplicity, are grouped).
- Track material also shows promise as a driver
- Clustering does not show much promise in improving model performance
## Modeling
- The best performing model was a simple Linear Regression
- Final model performed 33% better than the baseline

# Recommendations:
- Collect more observations on coasters
- Indicating whether a coaster is launched or lift-hill powered would improve speed as a metric
- Update data set to include more current coasters. 