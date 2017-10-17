'''
Random Forest classifier implementation. One class represents a single node in
a decision tree. The links between these classes define the tree. These can
also be combined within another class to create a random forest. These classes
were implemented by hand instead of using a package in order to learn the
algorithm
'''

#
#   Copyright (C) 2017   Ezra Erb
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License version 3 as published
#   by the Free Software Foundation.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#   I'd appreciate a note if you find this program useful or make
#   updates. Please contact me through LinkedIn or github (my profile also has
#   a link to the code depository)
#
import itertools
import numpy as np
import scipy.stats as stats

def get_entropy(categories):
    '''
    Compute the entropy of a given distribution of categories
    '''

    # In information theory, entropy is defined as the negative of the weighted
    # average of the log2 of the probability a given element is in each
    # category.

    # force this calculation as a float
    probabilities = categories.astype(np.float64) / categories.sum()

    # Entropy of elements with zero probability is zero. This test is needed
    # to avoid log2 of zero, which is negative infinity. Unfortunately, 'where'
    # calculates both results before doing the test, so need to run it inside
    # this block to suppress the warnings it would otherwise issue.
    # http://stackoverflow.com/questions/29950557/ignore-divide-by-0-warning-in-python
    with np.errstate(divide='ignore', invalid='ignore'):
        weighted_entropies = np.where(probabilities > 0,
                                      np.log2(probabilities) * probabilities, 0.0)

    return -weighted_entropies.sum()

def calc_information_gain(left_side_values, right_side_values, overall_entropy,
                          overall_size):
    '''
    Find the information gain for a set of potential splits. The values must
    have the same number of entries or this method fails
    '''

    # The algorithm used to calculate the information gain is the classic
    # TF-IDF algorithm. It uses the ratio of information gain from the split to
    # the maximum gain possible from the split. The maximum possible gain is
    # one where every piece of the split contains entries from a single
    # category, so need to form another array of the split sizes
    split_counts = np.stack([np.sum(left_side_values, 1), np.sum(right_side_values, 1)], -1)

    # Calculate the entropy of each array and the overall size array.
    left_side_entropy = np.apply_along_axis(get_entropy, 1, left_side_values)
    right_side_entropy = np.apply_along_axis(get_entropy, 1, right_side_values)
    max_split_entropy = np.apply_along_axis(get_entropy, 1, split_counts)

    # Information gain is the entropy of the combined data minus the weighted
    # average of the entropy of the split pieces.

    # Scale left and right side entropies
    left_side_entropy = (left_side_entropy * split_counts.T[0]) / overall_size
    right_side_entropy = (right_side_entropy * split_counts.T[1]) / overall_size
    information_gain = overall_entropy - left_side_entropy - right_side_entropy
    # Now divide by the maximum possible gain for each split to normalize
    information_gain /= max_split_entropy
    return information_gain

def assemble_potential_splits(data_set, filter_columns, category_counts):
    '''
    Given a set of training data mapped to categoies, find where that data can
    potentially be split, and assemble data for each split
    '''

    split_values = []
    left_side_values = []
    right_side_values = []
    drop_columns = []

    for column in filter_columns:
        # Reduce the training set to just this column and the category, and
        # sort it. Note the use of transpose; rows are easier to index
        column_data = data_set.T[[column, -1]]
        sorted_indexes = np.lexsort((column_data[-1], column_data[0]))
        column_data = column_data[:, sorted_indexes]
        # Can only use this column to split if it has multiple values.
        if column_data[0][0] == column_data[-1][0]:
            # Any future split will also fail, so drop it.
            drop_columns.append(column)
        else:
            # Can only split at points where values are different and the
            # category changes. Nodes with a single category are filtered
            # before calling this routine, so guarenteed to find such a point.
            # If the category changes within a given value split at the next
            # value (this is unlikely to occur in practice)
            left_categories = np.zeros(category_counts.shape, category_counts.dtype)
            right_categories = category_counts.copy() # Modified
            last_value = column_data.T[0] # Seed with first row
            need_split = False
            for test_value in column_data.T:
                if test_value[1] != last_value[1]:
                    need_split = True
                if need_split and (test_value[0] != last_value[0]):
                    # Record the data of the potential split
                    # Note extra parentheses for a tupple
                    split_values.append((column, test_value[0]))
                    left_side_values.append(left_categories.copy())
                    right_side_values.append(right_categories.copy())
                    last_value = test_value
                    need_split = False

                # This entry is about to move from the right to the left of the
                # split, so update category counts for each half
                # NOTE: Tempting to count the entries and do a single update but
                # nparray accesses are fast enough it won't save much, and it
                # introduces more risk of the counts getting out of sync
                left_categories[(int)(test_value[1])] += 1
                right_categories[(int)(test_value[1])] -= 1

    return (split_values, left_side_values, right_side_values, drop_columns)

def find_split_point(training_data, filter_columns, category_counts):

    '''
    Given a set of training data and columns to consider, find the place to
    split the training data that maximizes information gain. Columns to
    consider in future splits are also returned.
    '''

    # To find the decision point for this tree, need to find the input field
    # and value that causes the largest decrease in entropy when the input
    # data is split there. Entropy values can be efficiently calculated in
    # bulk, so assemble a list of the category results of all possible splits

    # Need the entropy of the current set below, calculate it once
    overall_entropy = get_entropy(category_counts)

    # Need the overall number of entries too
    overall_size = category_counts.sum()

    (split_values, left_side_values, right_side_values, drop_columns) = \
        assemble_potential_splits(training_data, filter_columns, category_counts)

    # In very rare cases involving entries with the same value in all columns
    # to consider but different categories, split points may not be found. They
    # are so rare just bail now
    if len(split_values) == 0:
        return None

    if len(drop_columns) > 0:
        new_filter_columns = np.setdiff1d(filter_columns, drop_columns, True)
    else:
        new_filter_columns = filter_columns

    # Convert the final lists to NDarrays
    left_side_values = np.asarray(left_side_values)
    right_side_values = np.asarray(right_side_values)

    information_gain = calc_information_gain(left_side_values, right_side_values,
                                             overall_entropy, overall_size)

    # The split point is the point of maximum information gain. It gets paired
    # the the new filter columns for the retun value
    return (split_values[np.argmax(information_gain)], new_filter_columns)

def split_data_set(data_set, column, value):
    '''
    Split a set of data into two based on the column and value
    '''

    # http://stackoverflow.com/questions/21757680/python-separate-matrix-by-column-values
    left_data_set = data_set[data_set[:, column] < value, :]
    right_data_set = data_set[data_set[:, column] >= value, :]
    return (left_data_set, right_data_set)

def test_prediction_error(classifier, test_data):
    '''
    Given a set of observactions to classify with expected results, and a
    classifier, return the number that are misclassified
    '''

    # Its possible to have no observations. In this case treat the
    # misclassifiation rate as zero
    misclassification_count = 0
    if test_data.shape[0] > 0:
        predictions = np.apply_along_axis(classifier.classify_activity, 1, test_data)
        # Actual categories are the last column of the values
        # https://stackoverflow.com/questions/25490641/check-how-many-elements-are-equal-in-two-numpy-arrays-python
        misclassification_count = (test_data.T[-1] != predictions).sum()
    return misclassification_count

class RandomForest(object):  # pylint: disable=too-few-public-methods
    '''
    The entire random forest. This is basically a wrapper around the trees.
    Initialize with the training data and the number of trees in the forest
    '''
    def __init__(self, training_data, tree_count):
        # Ignore invalid data
        if tree_count < 1:
            print ('Tree count {} invalid; must have at least one tree. Set to '
                   'one'.format(tree_count))
            tree_count = 1

        self.trees = []

        # First, divide the samples into subsets. This is done by sampling
        # both samples and columns. The last one is the category and must
        # always be included. Rows and columns are chosen at random with
        # replacement so some of both get left out

        (full_rows, full_columns) = training_data.shape
        for dummy in itertools.repeat(None, tree_count):
            subspace_rows = np.unique(np.random.choice(full_rows, full_rows))

            # Get which columns to use for classification. Note how this
            # excludes the category in the last column
            subspace_columns = np.unique(np.random.choice(full_columns - 1, full_columns - 1))

            # Filter the wanted rows
            subspace_data = training_data[subspace_rows]

            # Divide the training data into training and pruning data. Reserve
            # one thid for pruning
            subspace_training_size = subspace_data.shape[0] / 3
            validation_data = subspace_data[:subspace_training_size]
            subspace_training_data = subspace_data[subspace_training_size:]
            self.trees.append(TreeNode(subspace_training_data, validation_data,
                                       subspace_columns, 0))

    def classify_activity(self, activity):
        '''
        Classify a given set of activity readings
        '''

        # To classify an activity with the forest, classify it with each tree
        # and return whatever category is chosen the most
        results = [tree.classify_activity(activity) for tree in self.trees]

        # Mode returns a tupple with the mode as the first value, in the same
        # dimension as the input
        return stats.mode(results)[0][0]

    def __str__(self):
        return '\n\n'.join([str(tree) for tree in self.trees])

    def __repr__(self):
        return self.__str__()

class TreeNode(object):   # pylint: disable=too-few-public-methods
    '''
    One node within a decision tree. It is implemented using a composite variant
    pattern where a given node can be interior or a leaf based on the data it
    contains. This pattern is used so they can be easily converted while
    constructing and pruning
    '''

    def __init__(self, training_data, validation_data, filter_columns, level):

        # Find counts of current categories in training data. Category is
        # always the last column. Stored as a float to match the other entries,
        # so need to cast
        category_counts = np.bincount(training_data.T[-1].astype(int))

        # Set the category of the node to the most popular category. If more
        # than one, take the first
        self.category = np.argmax(category_counts)

        # Record the size of the training data for this node. This is useful
        # for evaluating how well a learned tree split the data
        self.data_size = training_data.shape[0]

        self.level = level
        self.split_data = None

        # If all entries are in one category, have a leaf and can stop now
        if np.count_nonzero(category_counts > 0) == 1:
            return

        # If the node has no validation data, any created subtrees will be
        # pruned as overfitting, so declare this a leaf and stop
        if validation_data.shape[0] == 0:
            return

        # Set the split column and value for this node
        split_point = find_split_point(training_data, filter_columns, category_counts)
        if split_point is None:
            # No place to split exists. Bail now
            return

        filter_columns = split_point[1]
        self.split_data = {"column": split_point[0][0], "value": split_point[0][1]}
        # Divide the entries based on the split point. Order does not matter
        split_training_data = split_data_set(training_data,
                                             self.split_data["column"],
                                             self.split_data["value"])
        split_validation_data = split_data_set(validation_data,
                                               self.split_data["column"],
                                               self.split_data["value"])

        # Construct nodes from the left and right side data
        self.split_data["left_branch"] = TreeNode(split_training_data[0],
                                                  split_validation_data[0],
                                                  filter_columns, level + 1)
        self.split_data["right_branch"] = TreeNode(split_training_data[1],
                                                   split_validation_data[1],
                                                   filter_columns, level + 1)

        # Tree algorithms have a major problem with overfitting. The classic
        # solution is to find the classification error for each lower node,
        # and compare it to the error if this node were a leaf. If the latter
        # has a lower error rate, creating child nodes causes overfitting and
        # they should be pruned
        overall_error_count = np.count_nonzero(validation_data.T[-1] != self.category)
        split_error_count = test_prediction_error(self.split_data["left_branch"],
                                                  split_validation_data[0])
        split_error_count += test_prediction_error(self.split_data["right_branch"],
                                                   split_validation_data[1])

        if overall_error_count <= split_error_count:
            # Clear subtrees
            self.split_data = None

    def classify_activity(self, activity):
        '''
        Classify the activity based on this node. If a leaf it will return the
        category, otherwise it will call the appropriate branch
        '''

        # If the node has no splitting data, just return its category
        if self.split_data is None:
            return self.category
        # If the split column is not in the activity data, this is an error.
        # Return a negative category
        elif self.split_data["column"] >= activity.shape:
            return -1
        elif activity[self.split_data["column"]] < self.split_data["value"]:
            return self.split_data["left_branch"].classify_activity(activity)
        else:
            return self.split_data["right_branch"].classify_activity(activity)

    def __str__(self):
        # To enable matching up nodes in a big tree, add chars to show the
        # levels. Since the length varies, assemble as a list of strings and
        # join them
        output = []

        if self.level > 0:
            if self.level > 1:
                output += ['|'] * (self.level - 1)
            output.append('+')
        if self.split_data is None:
            output += ["Category: ", str(self.category), " Final data count: ", str(self.data_size)]
        else:
            output += ["Split col ", str(self.split_data["column"]), " at "]
            output += [str(self.split_data["value"]), "\n"]
            output += [str(self.split_data["left_branch"]), "\n"]
            output += [str(self.split_data["right_branch"])]
        return ''.join(output)

    def __repr__(self):
        return self.__str__()
