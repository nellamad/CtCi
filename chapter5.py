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


class Test(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
