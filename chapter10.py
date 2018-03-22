from collections import defaultdict, namedtuple
import random
from contextlib import contextmanager
from io import StringIO
import sys
import unittest


# 10.1 Sorted Merge: Given two sorted arrays, merge the 2nd into the 1st (assume that
# the first has enough buffer room at the end
def sorted_merge(a, b, m, n):
    last_merged = len(a) - 1
    for i in range(last_merged, -1, -1):
        if n > 0:
            if m > 0 and a[m - 1] > b[n - 1]:
                a[last_merged] = a[m - 1]
                m -= 1
            else:
                a[last_merged] = b[n - 1]
                n -= 1
            last_merged -= 1


# 10.2 Group Anagrams: Sort an array of strings so that all the anagrams are next to
# each other
def anagram_sort(strings):
    anagrams = defaultdict(list)

    for s in strings:
        anagram = ''.join(sorted(s))
        anagrams[anagram].append(s)

    return [s for a in anagrams.values() for s in a]


# 10.3 Search in a rotated array: Find a given element in sorted array of integers that
# has been rotated
def rotated_search_rec(rotated, start, end, target):
    if rotated is None:
        return None
    if start > end:
        return -1

    mid = (start + end) // 2

    if rotated[mid] == target:
        return mid
    if rotated[start] == target:
        return start
    if rotated[end] == target:
        return end

    if rotated[start] < rotated[mid]:
        if rotated[start] < target < rotated[mid]:
            return rotated_search_rec(rotated, start, mid - 1, target)
        else:
            return rotated_search_rec(rotated, mid + 1, end, target)
    elif rotated[mid] < rotated[end]:
        if rotated[mid] < target < rotated[end]:
            return rotated_search_rec(rotated, mid + 1, end, target)
        else:
            return rotated_search_rec(rotated, start, mid - 1, target)
    elif rotated[start] == rotated[mid]:
        if rotated[mid] != rotated[end]:
            return rotated_search_rec(rotated, mid + 1, end, target)
        else:
            left_result = rotated_search_rec(rotated, start, mid + 1, target)
            if left_result != -1:
                return left_result
            else:
                return rotated_search_rec(rotated, mid + 1, end, target)

    return -1


def rotated_search(rotated, target):
    return rotated_search_rec(rotated, 0, len(rotated) - 1, target)


# 10.4 Sorted Search, No Size: Given an instance of "Listy", an arraylist class with no size method and containing
# only positive integers, search for a given element.  A Listy will return -1 if given an invalid index.
class Listy(list):
    def __getitem__(self, key):
        try:
            return list.__getitem__(self, key)
        except IndexError:
            return -1


def sorted_search_no_size(listy, target):
    if listy is None:
        return None
    if listy[0] == -1:
        return -1
    if listy[1] == -1:
        return 0 if listy[0] == target else -1

    def binary_length_search(a, target):
        def binary_length_search_rec(a, start, step, target):
            if a[start] == -1:
                return start

            end = start + step
            if a[end] == -1:
                return binary_length_search_rec(a, start + 1 + (step // 2), 1, target)
            else:
                return end if target < a[end] else binary_length_search_rec(a, start, step * 2, target)

        return binary_length_search_rec(a, 0, 1, target)

    def binary_search_rec(a, start, end, target):
        if a is None:
            return None
        if start > end:
            return -1

        mid = (start + end) // 2
        if a[mid] == target:
            return mid
        elif a[mid] > target:
            return binary_search_rec(a, start, mid - 1, target)
        elif a[mid] < target:
            return binary_search_rec(a, mid + 1, end, target)

    end = binary_length_search(listy, target)
    return binary_search_rec(listy, 0, end, target)


# 10.5 Sparse Search: Given a sorted array of strings that is interspersed with empty strings, find a
# given string.
def sparse_search(strings, target):
    def sparse_search_rec(strings, start, end, target):
        if strings is None:
            return None
        if start > end:
            return -1

        mid = (start + end) // 2
        if strings[mid] == target:
            return mid

        for i in range(0, mid + 1):
            if strings[mid + i] != '':
                if target == strings[mid + i]:
                    return mid + i
                elif target < strings[mid + i]:
                    return sparse_search_rec(strings, start, mid - 1 - i, target)
                elif target > strings[mid + i]:
                    return sparse_search_rec(strings, mid + 1 + i, end, target)

                if target == strings[mid - i]:
                    return mid - i
                elif target < strings[mid - i]:
                    return sparse_search_rec(strings, start, mid - 1 - i, target)
                elif target > strings[mid - i]:
                    return sparse_search_rec(strings, mid + 1 + i, end, target)

        return -1

    return sparse_search_rec(strings, 0, len(strings) - 1, target)


# 10.8 Find Duplicates: Given an array of all numbers from 1 to N (max: 32000) and possible duplicates
# print all duplicate entries without being given N and given 4 kilobytes of memory

# NOTE: Ideal solution makes use of a bitarray which Python doesn't natively support, so the solution
# below will disregard the 4KB memory restriction
def print_duplicates(a):
    map_size = 16000
    lower_map = [False] * map_size # replacement for a bitarray
    upper_map = [False] * map_size

    def filter_lower(i):
        if i < map_size + 1 and not lower_map[i - 1]:
            lower_map[i - 1] = True
            return False
        else:
            return True

    def filter_upper(i):
        if i >= map_size and not upper_map[i - 1 - map_size]:
            upper_map[i - 1 - map_size] = True
            return False
        else:
            return True

    a = filter(filter_lower, a)
    a = filter(filter_upper, a)

    print(list(a))


# 10.9 Sorted Matrix Search: Given an MxN matrix where each row and column is sorted ascending, find
# a given element
Point = namedtuple('Point', ['x', 'y'])


def sorted_matrix_search(matrix, target):
    def sorted_matrix_search_rec(matrix, start, end, target):
        def point_in_matrix(matrix, point):
            return (len(matrix) > 0 and len(matrix[0]) > 0
                    and 0 <= point.x < len(matrix[0])
                    and 0 <= point.y < len(matrix))

        def binary_diagonal_insert_search(matrix, start, end, target):
            if ((start.x > end.x or start.y > end.y)
                    or not (point_in_matrix(matrix, start) and point_in_matrix(matrix, end))):
                return Point(-1, -1)

            mid_y = (start.y + end.y) // 2
            mid_x = start.x + (mid_y - start.y)
            mid = Point(mid_x, mid_y)

            if target == matrix[mid.y][mid.x]:
                return mid
            elif target > matrix[mid.y][mid.x]:
                if (mid.y + 1 == len(matrix)
                    or mid.x + 1 == len(matrix[0])
                        or matrix[mid.y + 1][mid.x + 1] > target):
                    return mid
                else:
                    return binary_diagonal_insert_search(matrix, Point(mid.x + 1, mid.y + 1), end, target)
            elif target < matrix[mid.y][mid.x]:
                if (mid.y == start.y
                    or mid.x == start.x
                        or matrix[mid.y - 1][mid.x - 1] < target):
                    return Point(mid.x - 1, mid.y - 1)
                else:
                    return binary_diagonal_insert_search(matrix, start, Point(mid.x - 1, mid.y - 1), target)

        mid = binary_diagonal_insert_search(matrix, start, end, target)

        if mid == Point(-1, -1) or start.y > mid.y or start.x > mid.x:
            return Point(-1, -1)
        if matrix[mid.y][mid.x] == target:
            return mid
        result = sorted_matrix_search_rec(matrix, Point(start.x, mid.y + 1), Point(mid.x, end.y), target)
        if result != (-1, -1):
            return result
        else:
            return sorted_matrix_search_rec(matrix, Point(mid.x + 1, start.y), Point(end.x, mid.y), target)

    if len(matrix) == 0 or len(matrix[0]) == 0:
        return Point(-1, -1)

    return sorted_matrix_search_rec(matrix, Point(0, 0), Point(len(matrix[0]) - 1, len(matrix) - 1), target)


# 10.10 Rank from Stream: Reading a stream of integers, implement track(x) and get_rank_of_number(x) to track and
# return the number of integers less than or equal to x that have been tracked.
class StreamReader:
    def __init__(self):
        self.sorted = []

    def binary_insert_search(self, x, start, end):
        if self.sorted == [] or end < start:
            return start
        if end - start == 0:
            return start if self.sorted[start] > x else start + 1

        mid = (start + end) // 2
        if x < self.sorted[mid]:
            return self.binary_insert_search(x, start, mid - 1)
        elif x >= self.sorted[mid]:
            return self.binary_insert_search(x, mid + 1, end)

    def track(self, x):
        self.sorted.insert(self.binary_insert_search(x, 0, len(self.sorted) - 1), x)

    def get_rank_of_number(self, x):
        return self.binary_insert_search(x, 0, len(self.sorted) - 1)


# 10.11 Peaks and Valleys: Given an array of integers, sort it into an alternating sequence of peaks and
# valleys.  Peak being an element greater than or equal to adjacent integers.  Valley being less than or equal.
def peak_sort(a):
    def swap(a, i, j):
        temp = a[i]
        a[i] = a[j]
        a[j] = temp

    if len(a) < 3:
        return
    for i in range(1, len(a) - 1, 2):
        if a[i - 1] >= a[i]:
            swap(a, i - 1, i)
        elif i + 1 < len(a) and a[i] <= a[i + 1]:
            swap(a, i, i + 1)


class Test(unittest.TestCase):
    @contextmanager
    def captured_output(self):
        new_out, new_err = StringIO(), StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = new_out, new_err
            yield sys.stdout, sys.stderr
        finally:
            sys.stdout, sys.stderr = old_out, old_err

    def test_sorted_merge(self):
        a = [5, 6, 7, 8]
        b = [1, 2, 3, 4]
        sorted_merge(a, b, 0, 4)
        self.assertEqual(a, [1, 2, 3, 4])

        a = [1, 2, 3, 4]
        b = []
        sorted_merge(a, b, 4, 0)
        self.assertEqual(a, [1, 2, 3, 4])

        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        b = [21, 22, 23, 24]
        sorted_merge(a, b, 6, 4)
        self.assertEqual(a, [1, 2, 3, 4, 5, 6, 21, 22, 23, 24])

    def test_anagram_sort(self):
        def verify_anagram_sorted(strings):
            roots = set()

            previous_root = None
            for s in strings:
                root = ''.join(sorted(s))
                if previous_root is not None and root != previous_root and root in roots:
                    return False
                roots.add(root)
                previous_root = root

            return True

        self.assertTrue(verify_anagram_sorted(anagram_sort([])))
        self.assertTrue(verify_anagram_sorted(anagram_sort(['dog'])))
        self.assertTrue(verify_anagram_sorted(anagram_sort(['abc', 'hello world', 'acb', 'bac', 'world hello'])))

    def test_rotated_search(self):
        self.assertEqual(-1, rotated_search([], 9))
        self.assertEqual(-1, rotated_search([6, 7, 8, 9, 0, 2, 3, 4, 5], 1))
        self.assertEqual(3, rotated_search([6, 7, 8, 9, 1, 2, 3, 4, 5], 9))
        self.assertEqual(8, rotated_search([6, 7, 8, 9, 1, 2, 3, 4, 5], 5))
        self.assertEqual(4, rotated_search([6, 7, 8, 9, 1, 2, 3, 4, 5], 1))
        self.assertEqual(5, rotated_search([6, 7, 8, 9, 1, 2, 3, 4, 5], 2))

    def test_sorted_search_no_size(self):
        self.assertEqual(-1, sorted_search_no_size(Listy([0, 1, 2, 3, 4]), 5))
        self.assertEqual(1, sorted_search_no_size(Listy([0, 1, 2, 3, 4]), 1))
        self.assertEqual(2, sorted_search_no_size(Listy([0, 1, 2, 3, 4]), 2))
        self.assertEqual(4, sorted_search_no_size(Listy([0, 1, 2, 3, 4]), 4))

    def test_sparse_search(self):
        self.assertEqual(-1, sparse_search([], 'cat'))
        self.assertEqual(-1, sparse_search([''], 'cat'))
        self.assertEqual(-1, sparse_search(['dog'], 'cat'))
        self.assertEqual(4, sparse_search(['at', '', '', '', 'bull', '', '', 'car', '', '', 'dad', '', ''], 'bull'))

    def test_print_duplicates(self):
        no_duplicates = [i for i in range(1, 32001)]
        random.shuffle(no_duplicates)

        with self.captured_output() as (out, err):
            print_duplicates(no_duplicates)
        output = eval(out.getvalue().strip())
        self.assertEqual([], output)

        duplicates = [random.randrange(1, 32001) for _ in range(0, 10)]
        with_duplicates = no_duplicates + duplicates
        with self.captured_output() as (out, err):
            print_duplicates(with_duplicates)
        output = eval(out.getvalue().strip())
        self.assertEqual(duplicates, output)

    def test_sorted_matrix_search(self):
        self.assertEqual(Point(-1, -1), Point(-1, -1))

        self.assertEqual(Point(-1, -1), sorted_matrix_search([], 5))
        self.assertEqual(Point(-1, -1), sorted_matrix_search([[1]], 0))

        matrix = [[1, 2, 3, 4, 5],
                  [6, 7, 8, 9, 10],
                  [11, 12, 13, 14, 15],
                  [16, 17, 18, 19, 20]]
        self.assertEqual(Point(4, 2), sorted_matrix_search(matrix, 15))

        matrix = [[1, 2, 3, 4, 5],
                  [6, 7, 8, 9, 10],
                  [11, 12, 13, 14, 15],
                  [16, 17, 18, 19, 20]]
        self.assertEqual(Point(3, 1), sorted_matrix_search(matrix, 9))

        matrix = [[1, 2, 3, 4, 5],
                  [6, 7, 8, 9, 10],
                  [11, 12, 13, 14, 15],
                  [16, 17, 18, 19, 20]]
        self.assertEqual(Point(0, 3), sorted_matrix_search(matrix, 16))

        matrix = [[1, 2, 3, 4, 5],
                  [6, 7, 8, 9, 10],
                  [11, 12, 13, 14, 15],
                  [16, 17, 18, 19, 20]]
        self.assertEqual(Point(3, 3), sorted_matrix_search(matrix, 19))

        matrix = [[0, 1, 2, 3, 4, 5],
                  [6, 7, 8, 9, 10, 11],
                  [12, 13, 14, 15, 16, 17],
                  [18, 19, 20, 21, 22, 23],
                  [24, 25, 26, 27, 28, 29]]
        self.assertEqual(Point(3, 2), sorted_matrix_search(matrix, 15))

    def test_stream_reader(self):
        reader = StreamReader()
        self.assertEqual(0, reader.get_rank_of_number(5))
        for i in range(0, 20):
            reader.track(i)
            self.assertEqual(i + 1, reader.get_rank_of_number(i))
        reader.track(4)
        reader.track(4)
        reader.track(11)
        self.assertEqual(7, reader.get_rank_of_number(4))
        self.assertEqual(15, reader.get_rank_of_number(11))

    def test_peak_sort(self):
        a = []
        peak_sort(a)
        self.assertEqual([], a)

        a = [0]
        peak_sort(a)
        self.assertEqual([0], a)

        def validate_peaks(a):
            for i in range(1, len(a) - 1, 2):
                self.assertLessEqual(a[i - 1], a[i])
                if i + 1 < len(a):
                    self.assertGreaterEqual(a[i], a[i + 1])

        a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        peak_sort(a)
        validate_peaks(a)

        a = [5, 3, 1, 2, 3]
        peak_sort(a)
        validate_peaks(a)
