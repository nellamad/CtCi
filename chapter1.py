import unittest


# 1.1 Is Unique: Check if a string has all unique characters
def is_unique(s):
    return len(set([x for x in s])) == len(s)


# 1.2 Check Permutation: Check if one string is a permutation of the other
def are_permutations(a, b):
    counts = {}
    for c in a:
        if c not in counts:
            counts[c] = 1
        else:
            counts[c] += 1
    for c in b:
        if c not in counts or counts[c] == 0:
            return False
        else:
            counts[c] -= 1
    return all([count == 0 for count in counts.values()])


# 1.3 URLify: Replace all spaces in a string with '%20'
def URLify(s):
    return s.replace(' ', '%20')


# 1.4 Palindrome Permutation: Check if a string is a permutation of a palindrome
def is_palindrome_permutation(s):
    counts = {}
    for c in s:
        if c not in counts:
            counts[c] = 1
        else:
            counts[c] += 1
    odd_counts = [count for count in counts.values() if count % 2 == 1]
    return (len(s) % 2 == 0 and len(odd_counts) == 0) or (len(s) % 2 == 1 and len(odd_counts) == 1)


# 1.5 One Away: Given 3 three operations (insert/remove/replace char), check if two strings are one or
# two operations away from each other
def is_one_away(s1, s2):
    longer = s1 if len(s1) >= len(s2) else s2
    shorter = s2 if len(s1) >= len(s2) else s1
    if len(longer) - len(shorter) > 1:
        return False
    difference_found = False
    difference = longer[:]
    for c in shorter:
        if c not in difference:
            if difference_found:
                return False
            difference_found = True
        difference = difference.replace(c, '', 1)
    return True


# 1.6 String Compression: Compress a string using the counts of repeated characters, e.g. aabcccccaaa -> a2b1c5a3
# If the compressed string is alrger than the original, return the original.  Charset is upper/lower a-z
def compress(s):
    compressed = []
    prevChar = ''
    count = 0
    for c in s:
        if c != prevChar:
            if prevChar != '':
                compressed.append(prevChar + str(count))
            prevChar = c
            count = 0
        count += 1
    compressed.append(prevChar + str(count))
    compressed = ''.join(compressed)
    return s if len(s) <= len(compressed) else compressed


# 1.7 rotateMatrix: Given an NxN matrix of 4 byte ints, rotate it by 90 degrees in place
def rotate_matrix(image):
    n = len(image)
    for level in range(0, int(n / 2)):
        for i in range(level, n - 1 - level):
            temp = image[level][i] # store top-left
            image[level][i] = image[n-1-i][level] # bottom-left to top-left
            image[n-1-i][level] = image[n-1-level][n-1-i] # bottom-right to bottom-left
            image[n-1-level][n-1-i] = image[i][n-1-level] # top-right to bottom-right
            image[i][n-1-level] = temp # top-left to top-right

# 1.8 Zero Matrix: Given an MxN matrix, change all elements in a column or row containing 0 to 0
def zero_matrix(matrix):
    if len(matrix) < 1:
        return
    
    n = len(matrix)
    m = len(matrix[0])
    zero_columns = [False] * m
    zero_rows = [False] * n
    for y in range(0, n):
        for x in range(0, m):
            if matrix[y][x] == 0:
                zero_columns[x] = True
                zero_rows[y] = True
    for y in range(0, n):
        for x in range(0, m):
            if zero_rows[y]:
                matrix[y] = [0] * m
                continue
            if zero_columns[x]:
                matrix[y][x] = 0


# 1.9 String Rotation: Given method isSubstring and strings s1, s2, check if s2 is a rotation of s1 using isSubstring once
# We'll use Python's built-in substring functionality
def is_rotation(s1, s2):
    return len(s1) == len(s2) and s1 in s2*2


class Test(unittest.TestCase):

    def test_isUnique(self):
        self.assertTrue(is_unique('Helo, Wrd!'))
        self.assertFalse(is_unique('Hello, World!'))

    def test_arePermutations(self):
        self.assertTrue(are_permutations('Hello, World!', 'oWHllo, lerd!'))
        self.assertFalse(are_permutations('Hello, ', 'World!'))

    def test_URLify(self):
        self.assertEqual(URLify('Hello, World!'), 'Hello,%20World!')

    def test_isPalindromePermutation(self):
        self.assertTrue(is_palindrome_permutation('Hello, World!Hel,Wrd!'))
        self.assertFalse(is_palindrome_permutation('Hello, World!'))

    def test_isOneAway(self):
        self.assertTrue(is_one_away('pale', 'ple'))
        self.assertTrue(is_one_away('pales', 'pale'))
        self.assertTrue(is_one_away('pale', 'bale'))
        self.assertFalse(is_one_away('pale', 'bake'))

    def test_compress(self):
        self.assertEqual(compress('aabcccccaaa'), 'a2b1c5a3')

    def test_rotateMatrix(self):
        image = [[1,2,3,4,5],
                 [16,17,18,19,6],
                 [15,24,25,20,7],
                 [14,23,22,21,8],
                 [13,12,11,10,9]]
        rotate_matrix(image)
        self.assertEqual(image, [[13, 14, 15, 16, 1], 
                                [12, 23, 24, 17, 2], 
                                [11, 22, 25, 18, 3], 
                                [10, 21, 20, 19, 4], 
                                [9, 8, 7, 6, 5]])

        matrix = [[0,2,3,4,5],
                 [16,17,18,19,6],
                 [15,24,25,0,7],
                 [14,23,22,21,8],
                 [13,12,11,10,9]]
        zero_matrix(matrix)
        self.assertEqual(matrix, [[0,0,0,0,0],
                             [0,17,18,0,6],
                             [0,0,0,0,0],
                             [0,23,22,0,8],
                             [0,12,11,0,9]])

    def test_isRotation(self):

        self.assertTrue(is_rotation('', ''))
        self.assertFalse(is_rotation('Hello,', 'World!'))
        self.assertTrue(is_rotation('waterbottle', 'terbottlewa'))


if __name__ == '__main__':
    unittest.main()


