from collections import namedtuple
from random import randrange
from contextlib import contextmanager
from io import StringIO
import sys
import unittest


# 8.1 Triple Step: Given a staircase of n steps and possible step sizes 1, 2, 3, implement a method to count how many
# possible ways there are to run up the stairs
def count_steps_rec(n, memo):
    if n < 0:
        return 0
    if n in memo:
        return memo[n]

    total = (count_steps_rec(n - 3, memo)
             + count_steps_rec(n - 2, memo)
             + count_steps_rec(n - 1, memo))

    memo[n] = total
    return total


def count_steps(n):
    return count_steps_rec(n, {0: 0, 1: 1, 2: 2, 3: 4})


# 8.2 Robots in a Grid: Grid with r rows, c columns, some cells impassable and a robot that can only move right or
# down.  Find a path from the top left to bottom right.
def get_path(c, r, off_cells):
    return get_path_rec(c, r, 0, 0, off_cells, '')


def get_path_rec(c, r, x, y, off_cells, prefix):
    if c < 1\
            or r < 1\
            or x >= c\
            or y >= r\
            or (x, y) in off_cells:
        return None
    if x + 1 == c and y + 1 == r:
        return prefix

    right_path = get_path_rec(c, r, x + 1, y, off_cells, prefix + 'r')
    if type(right_path) is str:
        return right_path
    down_path = get_path_rec(c, r, x, y + 1, off_cells, prefix + 'd')
    if type(down_path) is str:
        return down_path

    return None


# 8.3 Magic Index: Given a sorted array of distinct integers, find a magic index, if one exists. ie. find i | A[i] == i
def find_magic_index_rec(sorted, start, end):
    if sorted is None or len(sorted) == 0 or start > end:
        return -1

    median_index = start + ((end - start) // 2)
    median = sorted[median_index]
    if median == median_index:
        return median

    if median > median_index:
        return find_magic_index_rec(sorted, start, median_index - 1)
    if median < median_index:
        return find_magic_index_rec(sorted, median_index + 1, end)


def find_magic_index(sorted):
    return find_magic_index_rec(sorted, 0, len(sorted) - 1)


# 8.4 Power Set: Write a method to return all subsets of a set
def create_power_set(s):
    return create_power_set_rec(s, [[]])


def create_power_set_rec(s, subsets):
    if s is None:
        return None
    if len(s) == 0:
        return subsets

    new_sets = []
    for subset in subsets:
        new_set = subset.copy()
        new_set.append(s[0])
        new_sets.append(new_set)
    subsets += new_sets
    return create_power_set_rec(s[1:], subsets)


# 8.5 Recursive Multiply: Recursive function to multiply two positive integers without using *.  Can use addition,
# subtraction, and bit-shifting but try to minimize.
def slow_multiply(a, b):
    return slow_multiply_rec(max(a, b), min(a, b), 0)


def slow_multiply_rec(larger, smaller, total):
    if smaller == 0:
        return total

    return slow_multiply_rec(larger, smaller - 1, total + larger)


# 8.6 Towers of Hanoi: 3 towers of N disks.  Move disks from first tower to the last
class Stack:
    def __init__(self, elements=None):
        self.elements = elements if elements is not None else []

    def pop(self):
        return self.elements.pop()

    def push(self, element):
        self.elements.append(element)

    def __len__(self):
        return len(self.elements)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and other.elements == self.elements

    def __repr__(self):
        return 'Stack({0})'.format(self.elements)


def hanoi(source, buffer, destination):
    hanoi_rec(source, buffer, destination, len(source))


def hanoi_rec(source, buffer, destination, depth):
    if depth == 0:
        return
    if depth == 1:
        destination.push(source.pop())
        return

    hanoi_rec(source, destination, buffer, depth - 1)
    destination.push(source.pop())
    hanoi_rec(buffer, source, destination, depth - 1)


# 8.7 Permutations without duplicates: Write a method to compute all permutations of a string of unique characters
def compute_permutations_without_dups_rec(s, permutations):
    if s is None:
        return None
    if len(s) == 0:
        return permutations

    new_permutations = []
    new_char = s[0]
    for permutation in permutations:
        if permutation == '':
            new_permutations.append(new_char)
        else:
            for i in range(0, len(permutation) + 1):
                new_permutations.append(permutation[0:i] + new_char + permutation[i:])

    return compute_permutations_without_dups_rec(s[1:], new_permutations)


def compute_permutations_without_dups(s):
    return compute_permutations_without_dups_rec(s, [''])


# 8.8 Permutations with duplicates: Write a method to compute all permutations of a string whose characters are not
# necessarily unique.  Don't include duplicate permutations
def compute_permutations_rec(char_counts, prefix):
    if char_counts is None:
        return None
    if len(char_counts) == 0:
        return [prefix]

    new_permutations = []
    for char, char_count in char_counts.items():
        new_counts = char_counts.copy()
        new_counts[char] -= 1
        if new_counts[char] == 0:
            del new_counts[char]
        new_permutations += compute_permutations_rec(new_counts, prefix + char)

    return new_permutations


def compute_permutations(s):
    if s is None:
        return None
    char_counts = {}
    for c in s:
        char_counts[c] = char_counts[c] + 1 if c in char_counts else 1

    return compute_permutations_rec(char_counts, '')


# 8.9 Parents: Implement an algorithm to print all valid combinations of n pairs of parenthesis
def print_parens_rec(left_remaining, right_remaining, prefix):
    if right_remaining < left_remaining or left_remaining < 0 or right_remaining < 0:
        return

    if left_remaining == 0 and right_remaining == 0:
        print(prefix)
        return

    print_parens_rec(left_remaining - 1, right_remaining, prefix + '(')
    print_parens_rec(left_remaining, right_remaining - 1, prefix + ')')


def print_parens(n):
    print_parens_rec(n, n, '')


# 8.10 Paint Fill: Given a matrix of colours, a point and a new colour, implement the 'paint fill' function
def paint_fill_rec(screen, x, y, old_colour, new_colour):
    if (screen is None
            or len(screen) == 0
            or y < 0 or y >= len(screen)
            or x < 0 or x >= len(screen[0])
            or screen[y][x] != old_colour):
        return

    screen[y][x] = new_colour
    paint_fill_rec(screen, x + 1, y, old_colour, new_colour)
    paint_fill_rec(screen, x, y + 1, old_colour, new_colour)
    paint_fill_rec(screen, x - 1, y, old_colour, new_colour)
    paint_fill_rec(screen, x, y - 1, old_colour, new_colour)


def paint_fill(screen, x, y, new_colour):
    if (screen is None
            or len(screen) == 0
            or y < 0 or y >= len(screen)
            or x < 0 or x >= len(screen[0])):
        return

    paint_fill_rec(screen, x, y, screen[y][x], new_colour)


# 8.11 Coins: Given an infinite amount of quarters, dimes, nickles and pennies, calculate the number of ways
# possible to make n cents
def count_change_rec(n, denominations, memo):
    if n is None or denominations is None or len(denominations) == 0:
        return 0
    if n == 0 or len(denominations) == 1:
        return 1
    if (n, len(denominations)) in memo:
        return memo[(n, len(denominations))]

    if n < denominations[-1]:
        return count_change_rec(n, denominations[0:-1], memo)

    new_combinations = 0
    for i in range(0, (n // denominations[-1]) + 1):
        new_combinations += count_change_rec(n - (denominations[-1] * i),
                                             denominations[0:-1],
                                             memo)

    memo[(n, len(denominations))] = new_combinations
    return new_combinations


def count_change(n):
    return count_change_rec(n, [1, 5, 10, 25], {})


# 8.12 Eight Queens: Print all the ways of arranging 8 queens on an 8x8 chess board so that none of the
# queens share the same row, column or diagonal
def place_queens(n):
    place_queens_rec(n, 0, 0, [])


def place_queens_rec(n, x, y, placed):
    if n == 0:
        print(placed)
        return

    for (next_x, next_y) in next_point(x, y):
        if not is_colliding(next_x, next_y, placed):
            place_queens_rec(n - 1, next_x, next_y, placed + [(next_x, next_y)])


def next_point(x, y):
    for current_x in range(x, 8):
        yield (current_x, y)
    for current_y in range(y + 1, 8):
        for current_x in range(0, 8):
            yield (current_x, current_y)


def is_colliding(x, y, placed):
    for (queen_x, queen_y) in placed:
        if (x == queen_x
                or y == queen_y
                or x - y == queen_x - queen_y
                or x + y == queen_x + queen_y):
            return True

    return False


# 8.13 Stacks of Boxes: Given n boxes with width, height and height which cannot be rotated and may only
# be stacked on a strictly larger box (of every dimension), calculate the highest possible stack
Box = namedtuple('Box', ['width', 'depth', 'height'])


def compute_tallest_stack_rec(boxes, current_height, top):
    if boxes is None:
        return None
    valid_boxes = [box for box in boxes if top is None or (box.width < top.width
                                                           and box.depth < top.depth
                                                           and box.height < top.height)]
    if len(valid_boxes) == 0:
        return current_height

    max_height = current_height
    for box in valid_boxes:
        leftover = [b for b in valid_boxes.copy() if b != box]
        new_height = compute_tallest_stack_rec(leftover, current_height + box.height, box)
        max_height = max(max_height, new_height)

    return max_height


def compute_tallest_stack(boxes):
    return compute_tallest_stack_rec(boxes, 0, None)


# 8.14 Boolean Evaluation: Given a boolean expression consisting of 0, 1, &, |, ^ symbols and a desired
# boolean result, count the number of ways of parenthesizing the expression to to evaluate to the desired
# result
def pop_expression(expression):
    if expression is None:
        return None, None
    if expression == '':
        return '', None
    if expression[0] != '(':
        return expression[0], expression[1:]
    else:
        right_count = 1
        for i in range(1, len(expression)):
            if expression[i] == ')':
                right_count -= 1
            elif expression[i] == '(':
                right_count += 1
            if right_count == 0:
                return expression[1:i], expression[i + 1:]


def evaluate(expression):
    if expression is None or expression == '':
        return None
    if expression == '0':
        return False
    if expression == '1':
        return True

    (operand1, expression) = pop_expression(expression)
    operand1 = evaluate(operand1)
    while expression != '':
        (operator, expression) = pop_expression(expression)
        (operand2, expression) = pop_expression(expression)
        if operator == '&':
            operand1 = operand1 and evaluate(operand2)
        elif operator == '|':
            operand1 = operand1 or evaluate(operand2)
        elif operator == '^':
            operand2 = evaluate(operand2)
            operand1 = (operand1 or operand2) and not (operand1 and operand2)

    return operand1


def count_eval_rec(expression, result, start, end):
    ''' INCOMPLETE

    if expression is None or expression == '' or start > end:
        return 0

    total_count = 0
    for j in range(start, end - 3 if start == 0 else end, 2):
        incomplete_expression = expression[0:j] + '(' + expression[j:]
        for i in range(j + 4, end + 1, 2):
            complete_expression = incomplete_expression[0:i] + ')' + incomplete_expression[i:]
            if evaluate(complete_expression) == result:
                print((complete_expression, result, start, end))
            total_count += int(evaluate(complete_expression) == result) #\
               # if not (complete_expression[0] == '(' and complete_expression[-1] == ')') else 0
            total_count += count_eval_rec(complete_expression, result, j + 1, i)
            total_count += count_eval_rec(complete_expression, result, j + 3, i)
            total_count += count_eval_rec(complete_expression, result, i + 2, end)

    return total_count
    '''


def count_eval(expression, result):
    return count_eval_rec(expression, result, 0, len(expression) + 1)


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

    def test_count_steps(self):
        self.assertEqual(count_steps(-6), 0)
        self.assertEqual(count_steps(0), 0)
        self.assertEqual(count_steps(1), 1)
        self.assertEqual(count_steps(2), 2)
        self.assertEqual(count_steps(3), 4)
        self.assertEqual(count_steps(4), 7)
        self.assertEqual(count_steps(5), 13)

    def test_get_path(self):
        self.assertIsNone(get_path(-4, 4, {(1, 1)}))
        self.assertIsNone(get_path(0, 0, {}))
        self.assertIsNone(get_path(1, 1, {(0, 0)}))
        self.assertIsNone(get_path(5, 4, {(0, 1), (1, 0)}))
        self.assertEqual(get_path(1, 1, {}), '')

        def verify_path(c, r, x, y, off_cells, path):
            if path is None or (x, y) in off_cells:
                return False
            if x + 1 == c and y + 1 == r and path == '':
                return True

            if path[0] == 'r':
                return verify_path(c, r, x + 1, y, off_cells, path[1:])
            if path[0] == 'd':
                return verify_path(c, r, x, y + 1, off_cells, path[1:])

            return False

        off_cells = {(1, 2), (3, 3)}
        self.assertTrue(verify_path(5, 5, 0, 0, off_cells, get_path(5, 5, off_cells)))
        off_cells = {(5, 2), (6, 6), (1, 11)}
        self.assertTrue(verify_path(25, 17, 0, 0, off_cells, get_path(25, 17, off_cells)))

    def test_find_magic_index(self):
        non_magic = [x for x in range(-1, 198)]
        self.assertEqual(-1, find_magic_index(non_magic))
        for i in range(0, 100):
            magic = non_magic.copy()
            magic_index = randrange(0, len(magic))
            magic[magic_index] = magic_index
            magic[magic_index + 1:] = [x + 2 for x in magic[magic_index + 1:]]
            self.assertEqual(magic_index, find_magic_index(magic))

    def test_create_power_set(self):
        self.assertEqual(create_power_set([]), [[]])
        canonical_list = [x for x in range(0, 4)]
        power_set = create_power_set(canonical_list)
        self.assertEqual(power_set, [[],
                                     [0],
                                     [1], [0, 1],
                                     [2], [0, 2], [1, 2], [0, 1, 2],
                                     [3], [0, 3], [1, 3], [0, 1, 3], [2, 3], [0, 2, 3], [1, 2, 3], [0, 1, 2, 3]])

    def test_slow_multiply(self):
        self.assertEqual(12, slow_multiply(3, 4))
        self.assertEqual(0, slow_multiply(121, 0))
        self.assertEqual(15, slow_multiply(1, 15))

    def test_hanoi(self):
        buffer = Stack()
        destination = Stack()
        source = Stack([i for i in range(4, 0, -1)])
        hanoi(source, buffer, destination)
        self.assertEqual(source, Stack())
        self.assertEqual(buffer, Stack())
        self.assertEqual(destination, Stack([i for i in range(4, 0, -1)]))

        buffer = Stack()
        destination = Stack()
        source = Stack([i for i in range(10, 0, -1)])
        hanoi(source, buffer, destination)
        self.assertEqual(source.elements, [])
        self.assertEqual(buffer.elements, [])
        self.assertEqual(destination, Stack([i for i in range(10, 0, -1)]))

    def test_compute_permutations_without_dups(self):
        self.assertEqual(None, compute_permutations_without_dups(None))
        self.assertEqual([''], compute_permutations_without_dups(''))
        self.assertEqual({'xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx'}, set(compute_permutations_without_dups('xyz')))

    def test_computer_permutations(self):
        self.assertEqual(None, compute_permutations_without_dups(None))
        self.assertEqual([''], compute_permutations_without_dups(''))
        self.assertEqual({'xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx'}, set(compute_permutations_without_dups('xyz')))
        self.assertEqual({'xyy', 'yxy', 'yyx'}, set(compute_permutations('xyy')))

    def test_print_parens(self):
        with self.captured_output() as (out, err):
            print_parens(3)
        output = out.getvalue().strip()

        self.assertEqual({'((()))', '(()())', '()(())', '()()()', '(())()'}, set(output.split('\n')))

    def test_paint_fill(self):
        screen = [[1] * 11,
                  [1] * 11,
                  [1] * 11,
                  [1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 1],
                  [1] * 11]
        paint_fill(screen, 6, 3, 1)
        for row in screen:
            self.assertEqual([1] * 11, row)

        screen = [[1] * 11,
                  [1] * 11,
                  [1] * 11,
                  [1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 1],
                  [1] * 11]
        paint_fill(screen, 0, 0, 4)
        for i in [0, 1, 2, 4]:
            self.assertEqual([4] * 11, screen[i])
        self.assertEqual([4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 4], screen[3])

    def test_count_change(self):
        self.assertEqual(1, count_change(0))
        self.assertEqual(1, count_change(1))
        self.assertEqual(1, count_change(2))
        self.assertEqual(2, count_change(5))
        self.assertEqual(2, count_change(7))
        self.assertEqual(4, count_change(10))
        self.assertEqual(9, count_change(21))
        self.assertEqual(13, count_change(25))

    def test_place_queens(self):
        with self.captured_output() as (out, err):
            place_queens(8)
        placements = [eval(placement_str) for placement_str in out.getvalue().strip().split('\n')]
        # don't really know how many solutions there should be
        self.assertGreater(len(placements), 0)
        # but we can at least verify that each generated placement is valid
        for placement in placements:
            for (x, y) in placement:
                for (other_x, other_y) in [(a, b) for (a, b) in placement if (a, b) != (x, y)]:
                    self.assertTrue(x != other_x
                                    and y != other_y
                                    and x - y != other_x - other_y
                                    and x + y != other_x + other_y)

    def test_compute_tallest_stack(self):
        self.assertIsNone(compute_tallest_stack(None))
        self.assertEqual(compute_tallest_stack([Box(125, 10, 1)]),
                         1)
        self.assertEqual(compute_tallest_stack([Box(3, 3, 3),
                                                Box(2, 2, 2),
                                                Box(1, 1, 1)]),
                         6)
        self.assertEqual(compute_tallest_stack([Box(10, 5, 11),
                                                Box(5, 10, 11)]),
                         11)
        self.assertEqual(compute_tallest_stack([Box(10, 5, 11),
                                                Box(5, 10, 11),
                                                Box(4, 4, 4)]),
                         15)
        self.assertEqual(compute_tallest_stack([Box(10, 5, 11),
                                                Box(5, 10, 11),
                                                Box(4, 4, 4),
                                                Box(9, 4, 10)]),
                         21)

    def test_count_eval(self):
        #self.assertEqual(count_eval('1^0|0|1', False), 2)
        #self.assertEqual(count_eval('0&0&0&1^1|0', True), 10)


if __name__ == '__main__':
    unittest.main()
