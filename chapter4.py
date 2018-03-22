from sys import maxsize
from functools import reduce
from collections import OrderedDict
import random
import unittest


class GraphNode:
    def __init__(self, data, children=[]):
        self.data = data
        self.children = children


class BinaryNode:
    def __init__(self, data, left=None, right=None, parent=None):
        self.data = data
        self.left = left
        if self.left is not None:
            self.left.parent = self
        self.right = right
        if self.right is not None:
            self.right.parent = self
        self.parent = parent

    def __repr__(self):
        return 'BinaryNode({0})'.format(self.data)

    def __contains__(self, data):
        return (data == self.data or
            (self.left is not None and data in self.left) or
            (self.right is not None and data in self.right))

    def __eq__(self, other):
        return other is not None and self.data == other.data
    
    def __hash__(self):
        return hash(self.data)


class BinarySearchNode(BinaryNode):
    def __contains__(self, data):
        return (data == self.data or 
            (data in self.left if self.left is not None and data < self.data else False) or
            (data in self.right if self.right is not None and data > self.data else False))


class Queue:
    def __init__(self):
        self.entrance = []
        self.exit = []
        self.length = 0

    def enqueue(self, el):
        self.entrance.insert(0, el)
        self.length += 1

    def dequeue(self):
        self.check_and_fill_exit()
        if len(self.exit) > 0:
            popped = self.exit[0]
            del self.exit[0]
            self.length -= 1
            return popped

    def peek(self):
        self.check_and_fill_exit()
        if len(self.exit) > 0:
            return self.exit[0]

    def check_and_fill_exit(self):
        if len(self.exit) < 1:
            while len(self.entrance) > 0:
                popped = self.entrance[0]
                del self.entrance[0]
                self.exit.insert(0, popped)

    def __len__(self):
        return self.length


# 4.1 Route Between Nodes: Given a directed graph, design an algorithm
# to find out whether there is a route between two nodes
def path_exists(a, b):
    def breadth_first_search(a, b):
        if a is None or b is None:
            return None
        to_visit = Queue()
        visited = set()
        to_visit.enqueue(a)
        while len(to_visit) > 0:
            visiting = to_visit.dequeue()
            if visiting == b:
                return True
            if visiting not in visited:
                for child in visiting.children:
                    to_visit.enqueue(child)
            visited.add(visiting)
        return False

    return breadth_first_search(a, b) or breadth_first_search(b, a)


# 4.2 BST from array: Given a sorted-increasing array with unique integers,
# create a BST with minimal height
def create_bst(arr):
    def create_bst_rec(arr, start, end):
        if len(arr) < 1 or start < 0 or end > len(arr) - 1:
            print('Returning empty BST from arr {0} {1}-{2}'.format(arr, start, end))
            return None
        bst_length = end - start + 1
        if start == end:
            return BinaryNode(arr[start])
        elif bst_length == 2:
            return BinaryNode(
                arr[start],
                BinaryNode(arr[end]) if arr[end] <= arr[start] else None,
                BinaryNode(arr[end]) if arr[end] > arr[start] else None)
        else:
            median = start + int(bst_length / 2)
            return BinaryNode(
                arr[median],
                create_bst_rec(arr, start, median - 1),
                create_bst_rec(arr, median + 1, end))

    return create_bst_rec(arr, 0, len(arr) - 1)


# 4.3 List of Depths: Given a binary tree, create linked lists of nodes at each depth
def get_depths(head):
    if head is None:
        return None
    visited = set()
    to_visit = Queue()
    to_visit.enqueue((head, 0))
    depths = []
    while len(to_visit) > 0:
        (visiting, depth) = to_visit.dequeue()
        if visiting not in visited:
            if visiting.left is not None:
                to_visit.enqueue((visiting.left, depth + 1))
            if visiting.right is not None:
                to_visit.enqueue((visiting.right, depth + 1))
        if depth == len(depths):
            depths.append(None)
        visiting.next = depths[depth]
        depths[depth] = visiting
        visited.add(visiting)
    return depths


# 4.4 Check Balanced: Check if a binary tree is balanced (heights of two subtrees of a node differ by at most 1)
def is_balanced(root):
    return is_balanced_rec(root) is not False


def is_balanced_rec(root):
    if root is None:
        return 0
    left_height = is_balanced_rec(root.left)
    right_height = is_balanced_rec(root.right)
    if left_height is False or right_height is False or abs(left_height - right_height) > 1:
        #print('Unbalanced subtrees {2} and {3} with heights {0} and {1}'.format(left_height, right_height, root.left, root.right))
        return False
    return 1 + max(left_height, right_height)


# 4.5 ValidateBST: Check if a binary tree is a binary search tree
def validate_bst(root):
    return validate_bst_rec(root, -maxsize - 1, maxsize)


def validate_bst_rec(root, min, max):
    if root is None:
        return True
    return (min < root.data < max
            and validate_bst_rec(root.left, min, root.data)
            and validate_bst_rec(root.right, root.data, max))


# 4.6 Successor: Find the inorder successor of a given node in a BST.  Nodes have a parent link
def get_inorder_successor(root):
    if root is None:
        return None
    if (root.right is None and
            root.parent is not None and
            root.parent.data > root.data):
        return root.parent
    current = root.right
    while current.left is not None:
        current = current.left
    return current


# 4.7 Build Order: Given a list of projects and a list of (dependency, dependant), find a build
# order that will allow the projects to be built.  Return an error if it is not possible.
def get_build_order(projects, dependencies):
    if projects is None:
        return None
    if dependencies is None:
        return list(OrderedDict.fromkeys(projects))
    build = []
    next_level = projects[:]
    while len(next_level) > 0:
        #print('next_level: {0}'.format(next_level))
        build.append(next_level)
        
        current_level = build[-1]
        next_level = list(OrderedDict.fromkeys([dependant for (dependency, dependant) in dependencies if dependency in current_level]))
        if current_level == next_level:
            raise BuildDependencyCycleError(current_level)

        build[-1] = [project for project in current_level if project not in next_level]

    return reduce(lambda final_build, level: final_build + level, build)


class Error(Exception):
    pass


class BuildDependencyCycleError(Error):
    def __init__(self, message):
        self.message = message


# 4.8 First Common Ancestor: Find the first common ancestor of two binary nodes.  Avoid storing
# additional nodes in a data structure
def first_common_ancestor(root, a, b):
    if root is None:
        return None
    
    a_left = root.left is not None and a in root.left
    a_right = root.right is not None and a in root.right
    b_left = root.left is not None and b in root.left
    b_right = root.right is not None and b in root.right

    if ((root.data == a and (b_left or b_right)) or
            (root.data == b and (a_left or a_right)) or
            (a_left and b_right) or
            (b_left and a_right)):
        return root

    if ((not a_left) and (not a_right) or
            (not b_left) and (not b_right)):
        return None

    return first_common_ancestor(
        root.left if a_left and b_left else root.right,
        a,
        b)


# 4.9 BST Sequences: A BST was created by inserting elements from an array (in order).  Given a BST
# with distinct elements, print all possible arrays that could have led to this tree
def bst_sequence_mix(leftSource, rightSources):
    # print('left: {0}, rights: {1}'.format(leftSource, rightSources))
    if leftSource is None or len(leftSource) == 0:
        return rightSources
    sources = []
    for rightSource in rightSources:
        if len(rightSource) == 0:
            sources.append(leftSource)
            continue
        for i in range(0, len(leftSource) + 1):
            prefix = leftSource[0:i] + [rightSource[0]]
            # print('prefix: {0}'.format(prefix))
            suffices = bst_sequence_mix(leftSource[i:], [rightSource[1:]])
            # print('suffices: {0}'.format(suffices))

            for suffix in suffices:
                sources.append(prefix + suffix)

    return sources


def get_sequence_arrays(root):
    if root is None:
        return [[]]

    left_sources = get_sequence_arrays(root.left)
    right_sources = get_sequence_arrays(root.right)
    #print('lefts: {0}, rights: {1}'.format(left_sources, right_sources))

    sources = []
    for leftSource in left_sources:
        for source in bst_sequence_mix(leftSource, right_sources):
            sources.append([root.data] + source)

    return sources


# 4.10 Check Subtree: Check if one binary tree is a subtree of another much larger binary tree
def check_subtree(root1, root2):
    def serialize_tree(root):
        if root is None:
            return 'X'
        return '{0},{1},{2}'.format(root.data,
                                    serialize_tree(root.left),
                                    serialize_tree(root.right))

    return serialize_tree(root2) in serialize_tree(root1)


# 4.11 Random Node: Implement an algorithm to retrieve a random node from a binary tree.  All nodes
# should be equally likely to be chosen.
class BinaryTree:
    def __init__(self):
        self.nodes = []

    def add(self, el):
        self.nodes.append(BinaryNode(el))

    def remove(self, el):
        self.nodes.remove(BinaryNode(el))

    def get_random(self):
        return random.choice(self.nodes)

    # For testing purposes purely
    def get_nodes(self):
        return self.nodes


# 4.12 Paths with Sum: Given a binary tree of integers, count the number of paths that sum to a
# given value.  Paths don't need to start at root or end at leaf but they must go downwards.
def count_paths_sum_brute(root, sum):
    if root is None:
        return 0
    return (int(root.data == sum) +
            count_paths_sum(root.left, sum - root.data) +
            count_paths_sum(root.right, sum - root.data))


# solution from CtCi
def count_paths_sum(root, sum):
    def count_paths_sum_rec(root, target_sum, running_sum, path_counts):
        if root is None:
            return 0

        running_sum += root.data
        total_paths = path_counts[running_sum - target_sum] if (running_sum - target_sum) in path_counts else 0

        if running_sum == target_sum:
            total_paths += 1

        if running_sum not in path_counts:
            path_counts[running_sum] = 1
        else:
            path_counts[running_sum] += 1
        total_paths += count_paths_sum_rec(root.left, target_sum, running_sum, path_counts)
        total_paths += count_paths_sum_rec(root.right, target_sum, running_sum, path_counts)
        path_counts[running_sum] -= 1

        return total_paths

    return count_paths_sum_rec(root, sum, 0, {})


class Test(unittest.TestCase):

    def test_pathExists(self):
        b = GraphNode('!')
        a = GraphNode('h', [GraphNode('e', 
                                    [GraphNode('l'), GraphNode('l'), GraphNode('o', 
                                                                                    [b])])])
        self.assertTrue(path_exists(a, b))
        self.assertTrue(path_exists(b, a))

    def test_createBST(self):
        arr = [x for x in range(0, 11)]
        BST = create_bst(arr)
        for i in range(0, 11):
            self.assertTrue(i in BST)
        self.assertTrue(-1 not in BST)
        self.assertTrue(11 not in BST)

    def test_getDepths(self):
        binary_tree = BinaryNode(1,
                        BinaryNode(2,
                            BinaryNode(4)),
                        BinaryNode(3,
                            BinaryNode(5,
                                BinaryNode(6),
                                BinaryNode(7,
                                    None,
                                    BinaryNode(8)))))
        depths = get_depths(binary_tree)
        self.assertIsNotNone(depths)
        self.assertTrue(
            depths[0].data == 1 and
            depths[0].next is None and
            depths[1].data == 3 and
            depths[1].next.data == 2 and
            depths[1].next.next is None and
            depths[2].data == 5 and
            depths[2].next.data == 4 and
            depths[2].next.next is None and
            depths[3].data == 7 and
            depths[3].next.data == 6 and
            depths[3].next.next is None and
            depths[4].data == 8 and
            depths[4].next is None and
            len(depths) == 5)

    def test_isBalanced(self):
        self.assertTrue(is_balanced(None))
        self.assertTrue(is_balanced(BinaryNode(1)))
        unbalanced_binary_tree = BinaryNode(1,
                                BinaryNode(2,
                                    BinaryNode(4)), 
                                BinaryNode(3,
                                    BinaryNode(5,
                                        BinaryNode(6),
                                        BinaryNode(7,
                                            None,
                                            BinaryNode(8)))))
        self.assertFalse(is_balanced(unbalanced_binary_tree))
        balanced_binary_tree = BinaryNode(1,
                                BinaryNode(2,
                                    BinaryNode(4,
                                        BinaryNode(9),
                                        BinaryNode(10)),
                                    BinaryNode(3)),
                                BinaryNode(5,
                                    BinaryNode(6),
                                    BinaryNode(7,
                                        None,
                                        BinaryNode(8))))
        self.assertTrue(is_balanced(balanced_binary_tree))

    def test_validateBST(self):
        balanced_binary_tree = BinaryNode(1,
                                BinaryNode(2,
                                    BinaryNode(4,
                                        BinaryNode(9),
                                        BinaryNode(10)),
                                    BinaryNode(3)),
                                BinaryNode(5,
                                    BinaryNode(6),
                                    BinaryNode(7,
                                        None,
                                        BinaryNode(8))))
        binary_search_tree = BinarySearchNode(10,
                                BinarySearchNode(4,
                                    BinarySearchNode(3)), 
                                BinarySearchNode(16,
                                    BinarySearchNode(12,
                                        BinarySearchNode(11),
                                        BinarySearchNode(13,
                                            None,
                                            BinarySearchNode(14)))))
        self.assertFalse(validate_bst(balanced_binary_tree))
        self.assertTrue(validate_bst(binary_search_tree))

    def test_getInorderSucessor(self):
        balanced_binary_search_tree = BinarySearchNode(1,
                                BinarySearchNode(2,
                                    BinarySearchNode(4,
                                        BinarySearchNode(9),
                                        BinarySearchNode(10)),
                                    BinarySearchNode(3)),
                                BinarySearchNode(5,
                                    BinarySearchNode(6),
                                    BinarySearchNode(7,
                                        None,
                                        BinarySearchNode(8))))
        binarySearchTree = BinarySearchNode(10, 
                                BinarySearchNode(4,
                                    BinarySearchNode(3)), 
                                BinarySearchNode(16,
                                    BinarySearchNode(12,
                                        BinarySearchNode(11),
                                        BinarySearchNode(13,
                                            None,
                                            BinarySearchNode(14)))))

        successor = get_inorder_successor(binarySearchTree)
        self.assertTrue(successor is not None and successor.data == 11)
        successor = get_inorder_successor(successor)
        self.assertTrue(successor is not None and successor.data == 12)
        successor = get_inorder_successor(successor)
        self.assertTrue(successor is not None and successor.data == 13)
        successor = get_inorder_successor(successor)
        self.assertTrue(successor is not None and successor.data == 14)

    def test_getBuildOrder(self):
        projects = ['a','b','c','d','e','f']
        dependencies = [('a','d'), ('f','b'), ('b','d'), ('f','a'), ('d','c')]
       
        self.assertIsNone(get_build_order(None, dependencies))
        
        build = get_build_order(projects, None)
        self.assertIsNotNone(build)
        self.assertEqual(len(build),len(set(build)))
        
        build = get_build_order(projects, dependencies)
        self.assertIsNotNone(build) 
        self.assertEqual(len(build), len(set(build)))
        self.assertTrue(all([build.index(dependency) < build.index(dependant) for (dependency,dependant) in dependencies]))
        
        dependencies = [('a','b'), ('b','c'), ('c', 'a')]
        self.assertRaises(BuildDependencyCycleError, get_build_order, projects, dependencies)

    def test_firstCommonAncestor(self):
        self.assertIsNone(first_common_ancestor(None, 4, 1))
        self.assertIsNone(first_common_ancestor(BinaryNode('a'), 'b', 'c'))
        balanced_binary_tree = BinaryNode(1,
                                BinaryNode(2,
                                    BinaryNode(4,
                                        BinaryNode(9),
                                        BinaryNode(10)),
                                    BinaryNode(3)),
                                BinaryNode(5,
                                    BinaryNode(6),
                                    BinaryNode(7,
                                        None,
                                        BinaryNode(8))))
        ancestor = first_common_ancestor(balanced_binary_tree, 3, 9)
        self.assertIsNotNone(ancestor)
        self.assertEqual(ancestor.data, 2)
        ancestor = first_common_ancestor(balanced_binary_tree, 1, 8)
        self.assertIsNotNone(ancestor)
        self.assertEqual(ancestor.data, 1)
        ancestor = first_common_ancestor(balanced_binary_tree, 4, 6)
        self.assertIsNotNone(ancestor)
        self.assertEqual(ancestor.data, 1)

    def test_getSequenceArrays(self):
        sequences = bst_sequence_mix([1], [[3]])
        self.assertEqual(len(sequences), 2)
        self.assertTrue([1, 3] in sequences and [3, 1] in sequences)
        sequences = get_sequence_arrays(BinarySearchNode(2, BinarySearchNode(1), BinarySearchNode(3)))
        self.assertEqual(len(sequences), 2)
        self.assertTrue([2, 1, 3] in sequences and [2, 3, 1] in sequences)
        sequences = get_sequence_arrays(
            BinarySearchNode(4,
                BinarySearchNode(2,
                    BinarySearchNode(1),
                    BinarySearchNode(3)),
                BinarySearchNode(5)))
        self.assertEqual(len(sequences), 8)
        self.assertTrue(
            [4, 5, 2, 1, 3] in sequences and
            [4, 2, 5, 1, 3] in sequences and
            [4, 2, 1, 5, 3] in sequences and
            [4, 2, 1, 3, 5] in sequences and
            [4, 5, 2, 3, 1] in sequences and
            [4, 2, 5, 3, 1] in sequences and
            [4, 2, 3, 5, 1] in sequences and
            [4, 2, 3, 1, 5] in sequences)
        sequences = get_sequence_arrays(
            BinarySearchNode(4,
                BinarySearchNode(2,
                    BinarySearchNode(1),
                    BinarySearchNode(3)),
                BinarySearchNode(5,
                    None,
                    BinarySearchNode(7))))
        self.assertEqual(len(sequences), 20)
        self.assertTrue(
            [4, 5, 7, 2, 1, 3] in sequences and
            [4, 5, 2, 7, 1, 3] in sequences and
            [4, 5, 2, 1, 7, 3] in sequences and
            [4, 5, 2, 1, 3, 7] in sequences and

            [4, 2, 5, 7, 1, 3] in sequences and
            [4, 2, 5, 1, 7, 3] in sequences and
            [4, 2, 5, 1, 3, 7] in sequences and

            [4, 2, 1, 5, 7, 3] in sequences and
            [4, 2, 1, 5, 3, 7] in sequences and

            [4, 2, 1, 3, 5, 7] in sequences and

            [4, 5, 7, 2, 3, 1] in sequences and
            [4, 5, 2, 7, 3, 1] in sequences and
            [4, 5, 2, 3, 7, 1] in sequences and
            [4, 5, 2, 3, 1, 7] in sequences and

            [4, 2, 5, 7, 3, 1] in sequences and
            [4, 2, 5, 3, 7, 1] in sequences and
            [4, 2, 5, 3, 1, 7] in sequences and

            [4, 2, 3, 5, 7, 1] in sequences and
            [4, 2, 3, 5, 1, 7] in sequences and

            [4, 2, 3, 1, 5, 7] in sequences)

    def test_checkSubtree(self):
        balanced_binary_tree = BinaryNode(1,
                                BinaryNode(2,
                                    BinaryNode(4,
                                        BinaryNode(9),
                                        BinaryNode(10)),
                                    BinaryNode(3)),
                                BinaryNode(5,
                                    BinaryNode(6),
                                    BinaryNode(7,
                                        None,
                                        BinaryNode(8))))
        subtree = BinaryNode(4,
                    BinaryNode(9),
                    BinaryNode(10))
        self.assertFalse(check_subtree(None, balanced_binary_tree))
        self.assertTrue(check_subtree(balanced_binary_tree, None))
        self.assertTrue(check_subtree(balanced_binary_tree, subtree))
        self.assertFalse(check_subtree(subtree, balanced_binary_tree))

    def test_randomNode(self):
        tree = BinaryTree()
        for i in range(0, 10):
            tree.add(i)

        random_state = random.getstate()
        random_nodes = [tree.get_random() for _ in range(0, 10)]
        random.setstate(random_state)
        self.assertTrue(random_nodes == [tree.get_random() for _ in range(0, 10)])
        self.assertFalse(random_nodes == [tree.get_random() for _ in range(0, 10)])

    def test_countPathsSum(self):
        binary_tree = BinaryNode(3,
                        BinaryNode(-1,
                            BinaryNode(4),
                            BinaryNode(2)),
                        BinaryNode(0,
                            BinaryNode(-6),
                            BinaryNode(-8)))
        self.assertEqual(count_paths_sum(binary_tree, 4), 2)


if __name__ == '__main__':
    unittest.main()


