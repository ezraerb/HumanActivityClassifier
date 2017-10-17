'''
Standard confusion matrix implementation. Implemented as a seperate class so
it is reusable across projects
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

class ConfusionMatrix(object):
    '''
    Initialize the class with the list of categories. The position of each
    category in the list must match the index used for that category within
    the classifier output
    '''
    def __init__(self, categories):
        self.categories = categories
        # Cached for higher perfomance
        self.category_count = len(categories)
        # The actual confusion matrix
        self.confusion_matrix = np.zeros((self.category_count, self.category_count), dtype=np.int)

    def add_result(self, expected, actual):
        '''
        Add A single result to the confusion matrix totals
        '''
        if (expected < 0) or (expected >= self.category_count):
            print ('result ignored. Expected value {} outside valid range 0 to'
                   ' {}'.format(expected, self.category_count))
        elif (actual < 0) or (actual >= self.category_count):
            print ('result ignored. Actual value {} outside valid range 0 to'
                   ' {}'.format(actual, self.category_count))
        else:
            self.confusion_matrix[actual][expected] += 1

    def report_stats(self):
        '''
        Calculate precision, recall, and balanced F statistic for each
        category given the classifier confusion matrix, and prints them
        '''

        # The important statistics for classification are precision and
        # recall. The former is the percentage of a predicted category items
        # that actually fall in that category. The latter is the precentaage
        # of items in a category that were predicted to be that category.
        # They are combined into an onverall statistic called the balanced F
        # statistic.
        #
        # This code calculates them category by category and then averages
        # those statistics to get overall values for the classifier. This
        # gives equal preference to the perfomance in each category

        # In a confusion matrix, the actual number in a category is the sum
        # of each row, the predicted number in a category is the sum of each
        # column, and the number correctly classified is the diagonal
        correctly_classified = np.diagonal(self.confusion_matrix).astype(np.float64)
        total_actual = np.sum(self.confusion_matrix, 1).astype(np.float64)
        total_predicted = np.sum(self.confusion_matrix, 0).astype(np.float64)

        # Handle the rare case of no test samples in a given category. This
        # implies the number correctly classified is zero, so the denonimator
        # in the calculations below does not matter. Note the less than one
        # test to handle float imprecision
        total_actual[total_actual < 1.0] = 1.0
        total_predicted[total_predicted < 1.0] = 1.0

        precision = correctly_classified / total_predicted
        recall = correctly_classified / total_actual

        # Avoid a divide by zero in the Fstatistic calculation by testing for
        # it up front. Since both precision and recall are non-negative, the
        # only way to get a zero value is for both to be zero, which implies
        # the numerator of the calculation will be zero. The denominator can
        # be anything in that case
        precision_recall_sum = precision + recall
        precision_recall_sum[precision_recall_sum < 1.0] = 1.0
        f_statistic = 2 * precision * recall / precision_recall_sum
        for cat, prec, rec, fstat in itertools.izip(self.categories, precision,
                                                    recall, f_statistic):
            print ('Category: {} Precision: {} Recall: {} Balanced F Statistic:'
                   ' {}'.format(cat, prec, rec, fstat))

            print ('Overall: Precision: {}  Recall: {} Balanced F Statistic:'
                   ' {}'.format(precision.mean(), recall.mean(), f_statistic.mean()))

    def __str__(self):
        ''' Pretty prints a confusion matrix '''

        # Want to print the matrix with columns and rows labeled. This
        # requires knowing the widths of each column. Assume there won't be
        # many of them so word wrap is not an issue. Also assume that the
        # sample count is low enough that the length of a printed value will
        # not exceed the longest column label
        max_word_length = 9 # "predicted"
        for category in self.categories:
            word_length = len(category)
            if word_length > max_word_length:
                max_word_length = word_length
        # Add two spaces between columns
        max_word_length += 2

        # Need the total number of columns. This is the width of the confusion
        # matrix plus 1
        output_columns = self.confusion_matrix.shape[1] + 1

        # For ease of assembling the output, create each row as a string,
        # and join then at the end
        row_output = []

        # Top label. It must line up in the center of the data columns. The
        # label will appear in the center of the specified width, so add
        # one extra column to the actual width to compensate for the extra
        # category column on the left
        row_output.append('{:^{width}}'.format('predicted',
                                               width=(output_columns + 1) * max_word_length))

        # Row with labels. Need to build as a list so each is justified within
        # its column, then join them
        column_output = ['{:^{width}}'.format('actual', width=max_word_length)]
        for category in self.categories:
            column_output.append('{:>{width}}'.format(category, width=max_word_length))
        row_output.append("".join(column_output))

        # To print the matrix need to iteate rows and labels and join the
        # column contents.
        # See http://stackoverflow.com/questions/17870612/printing-a-two-dimensional-array-in-python
        for category, values in itertools.izip(self.categories, self.confusion_matrix):
            column_output = ['{:>{width}}'.format(category, width=max_word_length)]
            for value in values:
                column_output.append('{:>{width}d}'.format(value, width=max_word_length))
            row_output.append("".join(column_output))

        # Assemble the final confusion matrix and return it
        return "\n".join(row_output)

    def __repr__(self):
        return self.__str__()
