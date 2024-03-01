import pandas as pd
import math
import random

# Set seed used for random number generator operations for reproducible results
random.seed(42) # Seed choice is abritrary, but 42 will be chosen for my testing

# Load data from training and test data files, assign column names, replace '?' with null
# Further preprocessing is done in other functions
def load_data_from_file(file_location):

    #  Give column attributes a name since they are not labeled in the data files
    column_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'Target']
    data = pd.read_csv(file_location, header=None, na_values='?', names=column_names, sep=',', 
                       dtype={'A1' : 'category', 'A4' : 'category', 'A5' : 'category', 
                              'A6' : 'category', 'A7' : 'category', 'A9' : 'category', 
                              'A10' : 'category', 'A12': 'category', 'A13' : 'category',
                              'Target' : 'category'})

    # Split attributes and target / label
    X = data.drop('Target', axis=1) 
    y = data['Target']

    return X, y
    
# Calculate medians for each attribute using training set data as per project instructions    
def calculate_medians_with_training_set(training_df):
    
    # Store medians for each column in a dictionary
    medians = {}

    for column in training_df.columns:
        if training_df[column].dtype == "object" or training_df[column].dtype == "category": # We check to see column is categorical
            
            # Sort values alphabetically, drop nulls, not removing duplicates
            values_sorted_alphabetically = sorted(training_df[column].dropna())
            
            # Get middle index
            middle_index = len(values_sorted_alphabetically) // 2

            # Get middle value ("median") for categorical data
            # When len of sorted values is even, we choose to use the second median.
            median = values_sorted_alphabetically[middle_index]

        else: # Data is numerical, we can simply use pandas's median function
            median = training_df[column].median()

        medians[column] = median

    return medians # Return dictionary

# Clean and preprocess data
def fill_in_missing_values(df, calculated_medians):
    for column in df.columns:

        # Check if this column has any missing values
        if pd.isnull(df[column]).any():

            # If so, fill them using the calculated median stored at the column key
            df[column] = df[column].fillna(calculated_medians[column])

    return df

# Split our data into sequential folds for cross-validation, number of folds can be passed as an argument alongside the 
# training data and labels.

# X represents our features in our dataset, expects a Pandas dataframe
# y is the target variable we want to predict, should be a pandas series
# number_of_folds is self explanatory, default is 10 as per project instructions 
def split_data_into_folds(X, y, number_of_folds=10):

    # This is a list of tuples, where each tuple will the training and validation sets for one fold.
    sequential_folds = [] # Store our sequential folds

    # Fold size, calculated by dividing the number of examples by the number of folds. 
    # A fold can't store part of an example so we use integer division.
    individual_fold_size = len(X) // number_of_folds

    # For the number of folds we want to create, keep iterating
    for fold_number in range(number_of_folds):

        # Get the starting index of current fold
        start_index = fold_number * individual_fold_size # First fold will start at index 0 and end at index 9 with fold of size of ten, second will start at 10, etc
        
        end_index = (fold_number + 1) * individual_fold_size # We will not include ending index, so we will set ending index to the starting index of the next fold
        
        # Create the validation set for the current fold by slicing the training data rows and labels using the start and end index
        
        X_fold_validation_set = X.iloc[start_index:end_index] # Remember end index is not included

        # Get the labels too
        y_fold_validation_set = y.iloc[start_index:end_index]

        # Training set for this fold should be everything that is not in the validation set
        # We get the rows before the start index of our validation set, and the rows starting (inclusive) from the end index
        X_fold_training_set = pd.concat([X.iloc[:start_index], X.iloc[end_index:]], axis=0) 

        # Do the same for the training set labels
        y_fold_training_set = pd.concat([y.iloc[:start_index], y.iloc[end_index:]], axis = 0) 

        # Store this fold as a tuple in our list of folds. Tuple contains the validation set and training set.
        sequential_folds.append(((X_fold_validation_set, y_fold_validation_set), (X_fold_training_set, y_fold_training_set)))

    # Return folds so they can be used for cross validation
    return sequential_folds

# The decision tree learning algorithm expects the features and labels together for the different examples, so we merge them back
def create_examples_dataframe(features, target):

    # Use a copy of the features dataframe as a base for the new data frame
    examples = features.copy() 

    # Assign target column to labels series
    examples['Target'] = target 

    return examples

# Get the most common target value from the passed subset of examples
def get_plurality_target_value(examples):
    
    # Use the pandas series mode method on the example dataframe's target series to get the most common label or labels (if tie)
    mode_values = examples['Target'].mode() # The mode function returns a series

    # If we have more than one most common target value
    if len(mode_values) > 1:

        # Convert series to list so we can use the random module's choice method to get a random mode to avoid bias. Seed is set at top of program.
        plurality_target_value = random.choice(mode_values.tolist()) 
    else: 
        plurality_target_value = mode_values.iloc[0] # There is only one mode so we assign that one to the plurality_target_value

    return plurality_target_value
    
# Check if each example after the current split with this attribute value leads to all examples having the same label
def all_have_same_classification_label(examples):
    
    # Get number of unique classifications. Based on current data set used, this can be + or -, representing credit card application accepted or rejected respectively.
    unique_classification_values = examples['Target'].unique() 

    # If there are more than one unique classifications, i.e both acceptions and rejections in this example subset, we need to return false
    if len(unique_classification_values) != 1:
        return False
    else:
        return True # Else, if there is one single classification label for all examples in this subset, return True

# Get entropy of a pandas dataset 'examples' with respect to the target values
def calculate_entropy(examples):

    if not isinstance(examples, pd.DataFrame):
        raise TypeError(f"Expected examples to be a pandas DataFrame, got {type(examples)} instead.")

    # Store the counts of each unique label in a dictionary
    unique_label_counts = {}

    # Increment the count of this label in the dictionary
    for label in examples['Target']:
        unique_label_counts[label] = unique_label_counts.get(label, 0) + 1

    # Get total number of examples in this dataset
    total_cardinality = len(examples)

    # Calculate the sum to be used in the entropy formula
    entropy_sum = 0

    for label_cardinality in unique_label_counts.values():

        # Calculate entropy of a single label as per entropy formula
        label_entropy = (label_cardinality / total_cardinality) * math.log2(label_cardinality / total_cardinality)

        # Add entropy of single label to entropy_sum to get entropy for entire dataset
        entropy_sum += label_entropy

    # Entropy must be positive so flip it as per the entropy formula
    dataset_entropy = -1 * entropy_sum

    return dataset_entropy # Return the final entropy of this examples dataset

# Calculate intrinisic value IV used in gain ratio formula
def calculate_intrinsic_value(examples, current_attribute):
    
    # Keep track of intrinsic value sum so far
    calculated_intrinsic_value = 0

    # For each unique value count, calculate its probability in current examples dataset and multiply that probability by the log2 of it.
    # After, add to split_info sum as per formula.
    for unique_value_count in examples[current_attribute].value_counts():

        unique_value_probability = unique_value_count / len(examples)

        # Only add unique value probability times its logarithm of base 2 if probability is non zero because we can't take a logarithm of 0.
        if unique_value_probability > 0:
            calculated_intrinsic_value += unique_value_probability * math.log2(unique_value_probability)
    
    # Distribute -1 to each term in the summation / take the opposite of the result as per the formula
    final_calculated_intrinsic_value = -1 * calculated_intrinsic_value

    # Return intrinsic value to be used for information gain ratio
    return final_calculated_intrinsic_value

# Calculate gini_index helper function, used for the get min gini index function
def calculate_gini_index(examples):

    # Gini index of an empty dataset is 0, return early, we do this because we can't divide by 0
    if len(examples) == 0:
        return 0

    # Keep track of the portion of the formula to be subtracted from 1
    gini_index_sum_portion = 0

    # For each unique target value, get its count
    for unique_target_count in examples['Target'].value_counts():

        # Get the probability of this target value in the current dataset
        unique_value_probability = unique_target_count / len(examples)

        # Add that probability squared to the sum portion
        gini_index_sum_portion += (unique_value_probability * unique_value_probability)

    # Finish calculating the gini index by subtracting the sum portion from 1
    gini_index = 1 - gini_index_sum_portion

    return gini_index
    
def get_max_information_gain_attribute(examples, attributes):

    if not isinstance(examples, pd.DataFrame):
        raise TypeError(f"Expected examples to be a pandas DataFrame, got {type(examples)} instead.")


    best_attribute = None
    best_information_gain = -float('inf') # Ensure any actual gain is greater than default value
    best_attribute_split_point = None # Keep track of whether best attribute is continous, if not None, it is continuous

    # Calculate entropy of full dataset without splitting on any attributes
    dataset_entropy = calculate_entropy(examples)
    full_dataset_length = len(examples)
    
    # For each attribute, we need to compute the information, and take the max
    for current_attribute in attributes:
        if examples[current_attribute].dtype == "object" or examples[current_attribute].dtype == "category": # Check if column is categorical
            
            # Sum used for calculating entropy of subset after being split on a attribute
            sum_to_subtract_from_dataset_entropy = 0

            # Need to calculate entropy of each unique attribute value
            for unique_value in examples[current_attribute].unique():

                # Get subset of examples where current attribute is set to current unique value
                examples_with_curr_attribute_as_unique_value = examples[examples[current_attribute] == unique_value]

                # Get length of this subset
                unique_value_dataset_length = len(examples_with_curr_attribute_as_unique_value)

                # Get entropy of subset with examples who have the current attribute set to the current unique value
                unique_value_entropy = calculate_entropy(examples_with_curr_attribute_as_unique_value)

                # Add the entropy of this attribute value times its cardinality divided by the total dataset cardinality
                sum_to_subtract_from_dataset_entropy += unique_value_entropy * (unique_value_dataset_length / full_dataset_length)

            # Calculate information gain per formula, used for calculating current attribute gain ratio    
            current_attribute_information_gain = dataset_entropy - sum_to_subtract_from_dataset_entropy

            # If below condition is met, we have a new best attribute to serve as our root node for our tree or subtree
            if current_attribute_information_gain > best_information_gain: 
                best_attribute = current_attribute
                best_information_gain = current_attribute_information_gain
                best_attribute_split_point = None # New best attribute is categorical, make sure best_split_point is none
        else: # Attribute is not categorical, therefore it is continous 

            # Sort examples by attribute value in ascending order
            examples_sorted_by_current_attribute = examples.sort_values(by=current_attribute)

            # Split point with the best information gain
            current_attribute_best_split_point = None
            current_attribute_best_split_point_information_gain = -float('inf')

            # Keep track of calculated split points to avoid redundant calculations
            calculated_split_points = set()

            # Use bi-partion and find split points
            for i in range(len(examples_sorted_by_current_attribute) - 1): # Stop at n - 1 to prevent index out of bound error

                # Calculate the midpoint of these two consecutive examples in terms of the current attribute value
                split_point = (examples_sorted_by_current_attribute.iloc[i][current_attribute] + examples_sorted_by_current_attribute.iloc[i + 1][current_attribute])

                # Make sure this split point is unique and has not already been calculated in the case of several duplicate continuous values
                if split_point in calculated_split_points:
                    continue

                # Split point is new so we we will now calculate the information gain from it
                calculated_split_points.add(split_point)

                # Split sorted examples in two datasets, one where examples have attribute <= split_point, and the other where it is greater.
                left_split_dataset = examples_sorted_by_current_attribute[examples_sorted_by_current_attribute[current_attribute] <= split_point]

                # Get the right split dataset
                right_split_dataset = examples_sorted_by_current_attribute[examples_sorted_by_current_attribute[current_attribute] > split_point]

                # Need to get entropy of both splits
                left_split_entropy = calculate_entropy(left_split_dataset)

                # Same for right split
                right_split_entropy = calculate_entropy(right_split_dataset)

                # Calculate weights for left and right split
                left_split_entropy_weight = len(left_split_dataset) / full_dataset_length

                # Same for right split
                right_split_entropy_weight = len(right_split_dataset) / full_dataset_length

                # Entropy of attribute given by current split point
                current_attribute_entropy_with_current_split = (left_split_entropy_weight * left_split_entropy) + (right_split_entropy_weight * right_split_entropy)

                # Calculate the information gain for current split
                current_attribute_information_gain_with_current_split = dataset_entropy - current_attribute_entropy_with_current_split

                # If this split point gives better a better gain ratio than the max seen so far
                if current_attribute_information_gain_with_current_split > current_attribute_best_split_point_information_gain:
                    current_attribute_best_split_point = split_point
                    current_attribute_best_split_point_information_gain = current_attribute_information_gain_with_current_split

            # After finding the split point which provides the most information gain for this attribute,
            # if the information gain found is better than any seen so far, we have a new best attribute to split on
            if current_attribute_best_split_point_information_gain > best_information_gain:
                best_attribute = current_attribute
                best_information_gain = current_attribute_best_split_point_information_gain
                best_attribute_split_point = current_attribute_best_split_point # New best attribute is continous, so set it's split point to a value other than None



    # Best attribute to serve as new root found using information gain
    return best_attribute, best_attribute_split_point

def get_max_gain_ratio_attribute(examples, attributes):
    
    best_attribute = None
    best_gain_ratio = -float('inf') # Ensure any actual gain is greater than default value
    best_split_point = None # Keep track of whether best attribute is continous, if not None, it is continuous

    # Calculate entropy of full dataset without splitting on any attributes
    dataset_entropy = calculate_entropy(examples)
    full_dataset_length = len(examples)
    
    # For each attribute, we need to compute the information gain ratio, and take the max
    for current_attribute in attributes:
        if examples[current_attribute].dtype == "object" or examples[current_attribute].dtype == "category": # Check if column is categorical
            
            # Sum used for calculating entropy of subset after being split on a attribute
            sum_to_subtract_from_dataset_entropy = 0

            # Need to calculate entropy of each unique attribute value
            for unique_value in examples[current_attribute].unique():

                # Get subset of examples where current attribute is set to current unique value
                examples_with_curr_attribute_as_unique_value = examples[examples[current_attribute] == unique_value]

                # Get length of this subset
                unique_value_dataset_length = len(examples_with_curr_attribute_as_unique_value)

                # Get entropy of subset with examples who have the current attribute set to the current unique value
                unique_value_entropy = calculate_entropy(examples_with_curr_attribute_as_unique_value)

                # Add the entropy of this attribute value times its cardinality divided by the total dataset cardinality
                sum_to_subtract_from_dataset_entropy += unique_value_entropy * (unique_value_dataset_length / full_dataset_length)

            # Calculate information gain per formula, used for calculating current attribute gain ratio    
            current_attribute_information_gain = dataset_entropy - sum_to_subtract_from_dataset_entropy

            # Get attribute's intrinsic value, also needed for calculating current attribute gain ratio. If 0, there is only one unique value for this current attribute in the dataset.
            intrinsic_value = calculate_intrinsic_value(examples, current_attribute)

            # Calculate the current attribute gain ratio if intrinsic value is not 0, i.e there is more than one unique attribute value in this current dataset
            if not(intrinsic_value == 0):
                current_attribute_gain_ratio = current_attribute_information_gain / intrinsic_value
            else: # There is only one unique value for this attribute in this dataset, and splitting on it yields no information gain, set current attribute's gain ratio to 0
                current_attribute_gain_ratio = 0

            # If new gain ratio is better than previously found max, we have a new best attribute to split on
            if current_attribute_gain_ratio > best_gain_ratio: 
                best_attribute = current_attribute
                best_gain_ratio = current_attribute_gain_ratio
                best_split_point = None # New best attribute is categorical, make sure best_split_point is none
        else: # Attribute is not categorical, therefore it is continous 

            # Sort examples by attribute value in ascending order
            examples_sorted_by_current_attribute = examples.sort_values(by=current_attribute)

            # Split point with the best information gain
            current_attribute_best_split_point = None
            current_attribute_best_split_point_gain_ratio = -float('inf')

            # Keep track of calculated split points to avoid redundant calculations
            calculated_split_points = set()

            # Use bi-partion and find split points
            for i in range(len(examples_sorted_by_current_attribute) - 1): # Stop at n - 1 to prevent index out of bound error

                # Calculate the midpoint of these two consecutive examples in terms of the current attribute value
                split_point = (examples_sorted_by_current_attribute.iloc[i][current_attribute] + examples_sorted_by_current_attribute.iloc[i + 1][current_attribute]) / 2

                # Make sure this split point is unique and has not already been calculated in the case of several duplicate continuous values
                if split_point in calculated_split_points:
                    continue

                # Split point is new so we we will now calculate the information gain from it
                calculated_split_points.add(split_point)

                # Split sorted examples in two datasets, one where examples have attribute <= split_point, and the other where it is greater.
                left_split_dataset = examples_sorted_by_current_attribute[examples_sorted_by_current_attribute[current_attribute] <= split_point]

                # Get the right split dataset
                right_split_dataset = examples_sorted_by_current_attribute[examples_sorted_by_current_attribute[current_attribute] > split_point]

                # If either split is empty, it's cardinality proportion would be 0 and we could not calculate the split's intrinsic value.
                # If we could not compute the intrinsic value, we could not compute the gain ratio because we divide the information gain by the intrinsic value to get it.
                # Set gain ratio to 0 for this split as it does not lead to a smaller example set
                if len(left_split_dataset) == 0 or len(right_split_dataset) == 0:
                    current_attribute_gain_ratio_with_current_split = 0
                else:
                    # Need to get entropy of both splits
                    left_split_entropy = calculate_entropy(left_split_dataset)

                    # Same for right split
                    right_split_entropy = calculate_entropy(right_split_dataset)

                    # Calculate weights for left and right split
                    left_split_entropy_weight = len(left_split_dataset) / full_dataset_length

                    # Same for right split
                    right_split_entropy_weight = len(right_split_dataset) / full_dataset_length

                    # Entropy of attribute given by current split point
                    current_attribute_entropy_with_current_split = (left_split_entropy_weight * left_split_entropy) + (right_split_entropy_weight * right_split_entropy)

                    # Calculate the information gain for current split
                    current_attribute_information_gain_with_current_split = dataset_entropy - current_attribute_entropy_with_current_split

                    # Get left split cardinality proportion of full dataset cardinality
                    left_split_cardinality_proportion = len(left_split_dataset) / full_dataset_length

                    # Get right split cardinality proportional of full dataset cardinality
                    right_split_cardinality_proportion = len(right_split_dataset) / full_dataset_length

                    # Calculate the intrinsic value for the current split 
                    current_attribute_intrinsic_value_with_current_split = -1 * (left_split_cardinality_proportion * math.log2(left_split_cardinality_proportion) + (right_split_cardinality_proportion * math.log2(right_split_cardinality_proportion)))

                    # Calculate gain ratio with current attribute's current split's information gain and intrinsic value
                    current_attribute_gain_ratio_with_current_split = current_attribute_information_gain_with_current_split / current_attribute_intrinsic_value_with_current_split

                # If this split point gives better a better gain ratio than the max seen so far
                if current_attribute_gain_ratio_with_current_split > current_attribute_best_split_point_gain_ratio:
                    current_attribute_best_split_point = split_point
                    current_attribute_best_split_point_gain_ratio = current_attribute_gain_ratio_with_current_split

            # After finding the split point which provides the best gain ratio for this attribute,
            # if the gain ratio found is better than any seen so far, we have a new best attribute to split on
            if current_attribute_best_split_point_gain_ratio > best_gain_ratio:
                best_attribute = current_attribute
                best_gain_ratio = current_attribute_best_split_point_gain_ratio
                best_split_point = current_attribute_best_split_point # New best attribute is continous, so set it's split point to a value other than None

    # Best attribute to serve as new root found using information gain ratio, best_split_point will be None if best attribute is attribute is categorical
    return best_attribute, best_split_point

# Get the attribute with the min gini index given passed examples dataset and attributes
def get_min_gini_index_attribute(examples, attributes):

    best_attribute = None
    best_gini_index = float('inf') # Ensure any actual gini index is less than default
    best_attribute_split_point = None # Keep track of whether best attribute is continous, if not None, it is continuous

    # Get length of full dataset for weight calculation
    full_dataset_length = len(examples)
    
    # For each attribute, we need to compute the information gain ratio, and take the max
    for current_attribute in attributes:
        if examples[current_attribute].dtype == "object" or examples[current_attribute].dtype == "category": # Check if column is categorical
            
            # Keep track of gini index for current attribute
            current_attribute_gini_index = 0

            # Calculate gini index of each attribute by summing gini index of each unique value for that attribute times its weight
            for unique_value in examples[current_attribute].unique():

                # Get subset of examples where current attribute is set to current unique value
                examples_with_curr_attribute_as_unique_value = examples[examples[current_attribute] == unique_value]

                # Get length of this subset where current attribute is set to current unique value
                unique_value_dataset_length = len(examples_with_curr_attribute_as_unique_value)

                # Calculate gini index for this attribute's current unique value
                unique_value_gini_index = calculate_gini_index(examples_with_curr_attribute_as_unique_value)

                # Calculate gini index for this attribute's current unique value's weight
                unique_value_gini_index_weight = unique_value_dataset_length / full_dataset_length

                # Add this unique value gini index times its weight to the current attribute gini index sum
                current_attribute_gini_index += (unique_value_gini_index * unique_value_gini_index_weight)
            
            # If current attribute gini index is less than previously found best, we have a new best attribute
            if current_attribute_gini_index < best_gini_index: 
                best_attribute = current_attribute
                best_gini_index = current_attribute_gini_index
                best_attribute_split_point = None # New best attribute is categorical, make sure best_split_point is none
        else: # Attribute is not categorical, therefore it is continous 

            # Sort examples by attribute value in ascending order
            examples_sorted_by_current_attribute = examples.sort_values(by=current_attribute)

            # Split point with the best gini index
            current_attribute_best_split_point = None
            current_attribute_best_split_point_gini_index = float('inf')

            # Keep track of calculated split points to avoid redundant calculations
            calculated_split_points = set()

            # Use bi-partion and find split points
            for i in range(len(examples_sorted_by_current_attribute) - 1): # Stop at n - 1 to prevent index out of bound error

                # Calculate the midpoint of these two consecutive examples in terms of the current attribute value
                split_point = (examples_sorted_by_current_attribute.iloc[i][current_attribute] + examples_sorted_by_current_attribute.iloc[i + 1][current_attribute]) / 2

                # Make sure this split point is unique and has not already been calculated in the case of several duplicate continuous values
                if split_point in calculated_split_points:
                    continue

                # Split point is new so we we will now calculate the information gain from it
                calculated_split_points.add(split_point)

                # Split sorted examples in two datasets, one where examples have attribute <= split_point, and the other where it is greater.
                left_split_dataset = examples_sorted_by_current_attribute[examples_sorted_by_current_attribute[current_attribute] <= split_point]

                # Get the right split dataset
                right_split_dataset = examples_sorted_by_current_attribute[examples_sorted_by_current_attribute[current_attribute] > split_point]

                # Calculate gini index of both splits
                left_split_dataset_gini_index = calculate_gini_index(left_split_dataset)

                # Do same for right split
                right_split_dataset_gini_index = calculate_gini_index(right_split_dataset)

                # Calculate weights for left and right split
                left_split_dataset_gini_index_weight = len(left_split_dataset) / full_dataset_length

                # Same for right split
                right_split_dataset_gini_index_weight = len(right_split_dataset) / full_dataset_length

                # Calculate gini index for current split point for current attribute
                # We do this by multiplying the cardinality of each split times its gini index, then summing these values
                current_split_point_gini_index = (left_split_dataset_gini_index_weight * left_split_dataset_gini_index) + (right_split_dataset_gini_index_weight * right_split_dataset_gini_index)

                # If this split point has a better (lower) gini index than the smallest seen so far, update the best for this attribute so far
                if current_split_point_gini_index < current_attribute_best_split_point_gini_index:
                    current_attribute_best_split_point = split_point
                    current_attribute_best_split_point_gini_index = current_split_point_gini_index

            # After finding the split point which provides the best gini index for this attribute
            # if the gini index found is better than any seen so far, we have a new best attribute to split on
            if current_attribute_best_split_point_gini_index < best_gini_index:
                best_attribute = current_attribute
                best_gini_index = current_attribute_best_split_point_gini_index
                best_attribute_split_point = current_attribute_best_split_point
    
    # Best attribute to serve as new root found using information gain ratio
    return best_attribute, best_attribute_split_point
     
def get_most_important_attribute(examples, attributes, importance_method):

    # best attribute to be root of new tree
    best_attribute = None

    # If decision tree is using Quinlan's C4.5
    if importance_method == "gain_ratio":
        best_attribute, best_attribute_split_point = get_max_gain_ratio_attribute(examples, attributes)
    elif importance_method == "gini_index":
        best_attribute, best_attribute_split_point = get_min_gini_index_attribute(examples, attributes)
    else: 
        best_attribute, best_attribute_split_point = get_max_information_gain_attribute(examples, attributes)

    # Debug
    if best_attribute == None:
        print(f"Best attribute {best_attribute} being returned with list of available attributes {attributes} with method {importance_method}")

    # Return best attribute found using the chosen method
    return best_attribute, best_attribute_split_point
    
# Train a decision tree based on the passed data, returns a decision tree model stored as a nested dictionary
def decision_tree_learning(examples, attributes, parent_examples, importance_method="information_gain"):
    
    # Base case for the recursion, if a split on an attribute leads to a branch having no examples, use plurality label for parent dataset examples. 
    if examples.empty == True:
        # If examples df is empty, and so is the parent examples dataframe, then the decision tree recieved no data to start with
        if parent_examples.empty:

            # We should raise an exception, there is an error in the data preprocessing pipeline
            raise ValueError("Attempted to train a decision tree from an empty dataset")
        else: # Parent examples df is not empty, use plurality label for parent dataset examples
            return get_plurality_target_value(parent_examples)
    elif all_have_same_classification_label(examples) == True: # Another base case, if a split leads to a pure subset, we stop splitting on this path.
        
        # Since all classifications are the same, we may simply return the label of the first example.
        common_classification_label = examples['Target'].iloc[0] 
        return common_classification_label
    elif len(attributes) == 0: # Final base case, if there are no more attributes left to split data on, use most common label using current subset of examples.
        return get_plurality_target_value(examples)
    else: # No stopping conditions are met, continue to split current example subset into further example subsets which can make more accurate predictions. 
        most_important_attribute, best_split_point_found = get_most_important_attribute(examples, attributes, importance_method) # Get most important attribute
        
        # Get new attributes list with the most important one removed
        # We do this because we don't want to split on the same attribute more than once on the same path.
        child_node_attributes = [attribute for attribute in attributes if attribute != most_important_attribute]

        # Create a new decision tree (dictionary) with the most important attribute at the root
        # The different unique values that attributes can be are keys in the nested dictionary for this attribute.
        # Each key in this attribute's nested dictionary is a potential value for this attribute.
        # It can either lead to a classification, or another nested dictionary where 
        # we can again split on another attribute which allows the tree to make more accurate classifications. 
        # Notice we have two conditions for the new root, one where the root attribute is categorical, 
        # and one where it is continuous. The else statement handles the continuous case.
        
        tree = {most_important_attribute: {}} 
        
        if best_split_point_found is None:
            for unique_value in examples[most_important_attribute].unique():

                # Get the examples in the examples df where the example's most important attribute is equal to the current unique value
                child_examples = examples[examples[most_important_attribute] == unique_value]

                # Create a subtree down this branch with the filtered down examples df, the current most important attribute removed, and the current examples
                # df passed as a parent examples df. We pass parent examples for the case that we get to an examples df with no entries, and we use the plurality value for the label of the parent.
                # This gives us a decent heuristic to use to make a classification if we reach a point where we lack sufficient data to make a more acccurate assessment.
                subtree = decision_tree_learning(child_examples, child_node_attributes, examples, importance_method=importance_method)
                
                # Add a branch to our tree with a label that A = current unique value, and the subtree or classification that arises from this spllit
                tree[most_important_attribute][unique_value] = subtree
        else: 
            # Get the subset to the left of the split point
            left_subset_examples = examples[examples[most_important_attribute] <= best_split_point_found]

            # Get the right subset examples as well
            right_subset_examples = examples[examples[most_important_attribute] > best_split_point_found]

            # Recurse to build the tree for both sides of the split point, proceeding with left first
            left_subtree = decision_tree_learning(left_subset_examples, child_node_attributes, examples, importance_method=importance_method)

            # Build right sub tree
            right_subtree = decision_tree_learning(right_subset_examples, child_node_attributes, examples, importance_method=importance_method)

            # Construct left subtree key for when the new instance's best attribute is <= split point
            left_subtree_key = f"<= {best_split_point_found}"

            # Construct right subtree key for when the new instance's best attribute is > split point
            right_subtree_key = f"> {best_split_point_found}"

            # Create new tree with branching paths to the subtrees
            tree[most_important_attribute] = {left_subtree_key : left_subtree, right_subtree_key : right_subtree}
    
    # Return tree to parent node, or back to main
    return tree

# Calculate attribute modes for a validation set, returns a dictionary
def calculate_attribute_modes(examples):

    attribute_modes = {}

    for current_attribute in examples.columns:

        # Mode function returns a series in Pandas, we 
        current_attribute_modes = examples[current_attribute].mode().tolist()
        attribute_modes[current_attribute] = current_attribute_modes

    return attribute_modes

def determine_attribute_types(examples):

    attribute_types = {}

    for attribute in examples.columns:
        if examples[attribute].dtype == 'object' or examples[attribute].dtype.name == 'category':
            attribute_types[attribute] = "Categorical"
        else:
            attribute_types[attribute] = "Continuous"
    
    return attribute_types
            
# Given a specific example, traverse the decision tree until we have a label
def predict_example(decision_tree, example, attribute_modes, attribute_types):
    
    # Start at root of our decision tree
    current_node = decision_tree

    # While current node is a dictionary, it means we have not yet taken a decision that leads us to a label. 
    # We need to go to the subtree of the next most important attribute, where the example's corresponding
    # attribute matches the split value.
    while isinstance(current_node, dict):

        # Get the attribute that was split on at this node
        for key in current_node: # Ends after one iteration
            most_important_attribute = key

        # Get the example's value for this most important attribute
        example_most_important_attribute_value = example[most_important_attribute]

        if attribute_types[most_important_attribute] == "Categorical":
            # Make sure attribute value exists as a subtree
            if example_most_important_attribute_value in current_node[most_important_attribute]:

                # Move to that subtree
                current_node = current_node[most_important_attribute][example_most_important_attribute_value]
            else: # Value doesn't exist try each mode

                found_value_subtree = False 

                for mode_value in attribute_modes[most_important_attribute]:
                    if mode_value in current_node[most_important_attribute]:
                        current_node = current_node[most_important_attribute][mode_value]
                        found_value_subtree = True
                        break

                # Could not find any valid subtree using an attribute's mode, randomly select an available subtree
                if not found_value_subtree:
                    available_subtrees = list(current_node[most_important_attribute].keys())
                    random_subtree_selected_key = random.choice(available_subtrees)
                    current_node = current_node[most_important_attribute][random_subtree_selected_key]
        elif attribute_types[most_important_attribute] == "Continuous":
            
            # Each item key represents condition and splitting threshold, and the value the subtree reached meeting that condition
            for condition, subtree in current_node[most_important_attribute].items():

                # Conditions look like "<= 25" so we get thresholds by splitting the string, and getting last element from resulting list
                check_type, threshold = condition.split(" ")

                # Convert threshold to a float
                threshold = float(threshold)

                if check_type == "<=":
                    if example_most_important_attribute_value <= threshold:
                        current_node = subtree
                elif check_type == ">":
                    if example_most_important_attribute_value > threshold:
                        current_node = subtree
        else:
            raise ValueError(f"Unrecognized attribute type for attribute {most_important_attribute}")
    
    # If current node is not a dictionary, we've reached our label, return it.
    return current_node


def predict_examples(decision_tree, examples_to_predict, attribute_types):

    # Calculate modes for current validation set incase an attribute value is unrecognized
    attribute_modes = calculate_attribute_modes(examples_to_predict)

    # Convert dataframe rows to a list of dictionaries so prediction operations on each example
    examples_list = examples_to_predict.to_dict('records')

    # Get the target label predictions for each example of this validation set
    y_validation_set_predictions = [predict_example(decision_tree, example, attribute_modes, attribute_types) for example in examples_list]

    # Return the predictions for this validation set
    return y_validation_set_predictions

# Calculate true and false positives, and true and false negatives
def calculate_confusion_matrix_values(y_validation_set_predictions, y_validation_set_labels):
    
    # Keep track of the positives
    true_positive = 0
    false_positive = 0
    
    # Keep track of the negatives
    false_negative = 0
    true_negative = 0 
    

    # Iterate through both lists at the same time
    for predicted, actual in zip(y_validation_set_predictions, y_validation_set_labels):
        if predicted == "+" and actual == "+": # Correctly predicted an application as accepted
            true_positive += 1
        elif predicted == "+" and actual == "-": # Falsely predicted an application as accepted
            false_positive += 1
        elif predicted == "-" and actual == "+": # Falsely predicted an application as rejected
            false_negative += 1
        elif predicted == "-" and actual == "-": # Correctly predicted an application as rejected
            true_negative += 1

    return true_positive, false_positive, false_negative, true_negative

def calculate_f1_score(y_validation_set_predictions, y_validation_set_actual_labels):
    # Calculate confusion matrix values for current fold
    true_positive, false_positive, false_negative, true_negative = calculate_confusion_matrix_values(y_validation_set_predictions, y_validation_set_actual_labels)

    # Calculate precision
    precision = true_positive / (true_positive + false_positive)

    # Calculate recall
    recall = true_positive / (true_positive + false_negative)

    # Calculate f1 score of fold
    f1_score = 2 / ((1 / precision) + (1 / recall))

    return f1_score

def perform_cross_validation(sequential_folds, attributes, attribute_types, importance_method="information_gain"):
    highest_f1_score = -float('inf')
    best_model = None

    for current_fold in sequential_folds:

        # Get validation data for current fold
        current_fold_X_validation_set, current_fold_y_validation_set = current_fold[0]

        # Get training data for current fold
        current_fold_X_training_set, current_fold_y_training_set = current_fold[1]

        # Create training examples dataframe from training data of current fold
        current_fold_training_examples_dataframe = create_examples_dataframe(current_fold_X_training_set, current_fold_y_training_set)
    
        # Train the model for this fold using the recursive decision tree algorithm
        current_fold_decision_tree = decision_tree_learning(current_fold_training_examples_dataframe, attributes, current_fold_training_examples_dataframe, importance_method)

        # Now that the model has been training, test on the validation set of this fold
        current_fold_y_validation_set_predictions = predict_examples(current_fold_decision_tree, current_fold_X_validation_set, attribute_types)

        current_fold_f1_score = calculate_f1_score(current_fold_y_validation_set_predictions, current_fold_y_validation_set)

        # If current fold model is better than any found so far, use it instead
        if current_fold_f1_score > highest_f1_score:
            highest_f1_score = current_fold_f1_score
            best_model = current_fold_decision_tree

    return best_model

# Load data training data
X_train, y_train = load_data_from_file('./data/training.data')

# Load test data
X_test, y_test = load_data_from_file('./data/test.data')

# Calculate medians using the training set
medians = calculate_medians_with_training_set(X_train)

# Clean training set
X_train_missing_values_filled = fill_in_missing_values(X_train, medians)

# Clean test set using same medians
X_test_missing_values_filled = fill_in_missing_values(X_test, medians)


'''
# For debugging and making sure preprocessing completed successfully

has_nulls = X_train_missing_values_filled.isnull().values.any()
print(f"Are there any null values in the Training DataFrame? {has_nulls}")

has_nulls = X_test_missing_values_filled.isnull().values.any()
print(f"Are there any null values in the Test DataFrame? {has_nulls}")
'''

# After cleaning data split it into folds
sequential_folds = split_data_into_folds(X_train_missing_values_filled, y_train)

# Define our attributes for the decision tree algorithm. We simply use the column names, A1, A2, A3..., etc.
# Use training features dataframe to get column names. Target is excluded as if it was included we would be using
# the outcome, what we are trying to predict, as part of the input.
attributes = list(X_train_missing_values_filled.columns) 

# Used for tree traversal for individual examples, dictionary storing attribute types
attribute_types = determine_attribute_types(X_train_missing_values_filled)

# Importance methods that can be used. Valid options are "information_gain", "gain_ratio", and "gini_index"
importance_methods = ["information_gain", "gain_ratio", "gini_index"]  

# Use cross validation to train and get the best information gain tree out of the best of the passed number of sequential folds
best_information_gain_model = perform_cross_validation(sequential_folds, attributes, attribute_types, importance_methods[0])

# Use information gain tree to predict labels for full test set
best_information_gain_y_test_predictions = predict_examples(best_information_gain_model, X_test, attribute_types)

# Calculate the f1 score of information gain tree on full test set
best_information_gain_f1_score = calculate_f1_score(best_information_gain_y_test_predictions, y_test)

print("Decision Tree - Best Information Gain Method - ID3 Algorithm Result")
print(f"F1 Score: {best_information_gain_f1_score}")

# Use cross validation to train and get the best gain ratio tree out of the best of the passed number of sequential folds
best_gain_ratio_model = perform_cross_validation(sequential_folds, attributes, attribute_types, importance_methods[1])

# Use the best gain ratio tree to predict labels for full test set
best_gain_ratio_y_test_predictions = predict_examples(best_gain_ratio_model, X_test, attribute_types)

# Calculate the f1 score of gain ratio tree on full test set
best_gain_ratio_f1_score = calculate_f1_score(best_gain_ratio_y_test_predictions, y_test)

print("Decision Tree - Best Gain Ratio Method - C4.5 Algorithm Result")
print(f"F1 Score: {best_gain_ratio_f1_score}")

# Use cross validation to train and get the best gini index tree out of the best of the passed number of sequential folds
best_gini_index_model = perform_cross_validation(sequential_folds, attributes, attribute_types, importance_methods[2])

# Use gini index tree to predict labels for full test set
best_gini_index_y_test_predictions = predict_examples(best_gini_index_model, X_test, attribute_types)

# Calculate the f1 score of gini index tree on full test set
best_gini_index_f1_score = calculate_f1_score(best_gini_index_y_test_predictions, y_test)

print("Decision Tree - Best Gini Index Method - Cart Algorithm Result")
print(f"F1 Score: {best_gini_index_f1_score}")