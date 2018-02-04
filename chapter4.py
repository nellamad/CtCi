from sys import maxsize
from functools import reduce
from collections import OrderedDict
import unittest
class GraphNode():
    def __init__(self, data, children=[]):
        self.data = data
        self.children = children

class BinaryNode():
    def __init__(self, data, left=None, right=None, parent=None):
        self.data = data
        self.left = left
        if self.left != None:
            self.left.parent = self
        self.right = right
        if self.right != None:
            self.right.parent = self
        self.parent = parent

    def __repr__(self):
        return 'BinaryNode({0})'.format(self.data)

    def __contains__(self, data):
        return (data == self.data or
            (self.left != None and data in self.left) or
            (self.right != None and data in self.right))
    

class BinarySearchNode(BinaryNode):
    def __contains__(self, data):
        return (data == self.data or 
            (data in self.left if self.left != None and data < self.data else False) or 
            (data in self.right if self.right != None and data > self.data else False))

class Queue():
    def __init__(self):
        self.entrance = []
        self.exit = []
        self.length = 0

    def enqueue(self, el):
        self.entrance.insert(0, el)
        self.length += 1

    def dequeue(self):
        self.checkAndFillExit()
        if len(self.exit) > 0:
            popped = self.exit[0]
            del self.exit[0]
            self.length -= 1
            return popped

    def peek(self):
        self.checkAndFillExit()
        if len(self.exit) > 0:
            return self.exit[0]

    def checkAndFillExit(self):
        if len(self.exit) < 1:
            while len(self.entrance) > 0:
                popped = self.entrance[0]
                del self.entrance[0]
                self.exit.insert(0, popped)

    def __len__(self):
        return self.length

# 4.1 Route Between Nodes: Given a directed graph, design an algorithm
# to find out whether there is a route between two nodes
def BreadthFirstSearch(a, b):
    if a == None or b == None:
        return None
    toVisit = Queue()
    visited = set()
    toVisit.enqueue(a)
    while len(toVisit) > 0:
        visiting = toVisit.dequeue()
        if visiting == b:
            return True
        if visiting not in visited:
            for child in visiting.children:
                toVisit.enqueue(child)
        visited.add(visiting)
    return False

def pathExists(a, b):
    return BreadthFirstSearch(a, b) or BreadthFirstSearch(b, a)

# 4.2 BST from array: Given a sorted-increasing array with unique integers,
# create a BST with minimal height
def createBST(arr):
    return createBSTRec(arr, 0, len(arr) - 1)

def createBSTRec(arr, start, end):
    if len(arr) < 1 or start < 0 or end > len(arr) - 1:
        print('Returning empty BST from arr {0} {1}-{2}'.format(arr, start, end))
        return None
    BSTLength = end - start + 1
    if start == end:
        return BinaryNode(arr[start])
    elif BSTLength == 2:
        return BinaryNode(
            arr[start],
            BinaryNode(arr[end]) if arr[end] <= arr[start] else None,
            BinaryNode(arr[end]) if arr[end] > arr[start] else None)
    else:
        median = start + int(BSTLength / 2)
        return BinaryNode(
            arr[median],
            createBSTRec(arr, start, median - 1),
            createBSTRec(arr, median + 1, end))

# 4.3 List of Depths: Given a binary tree, create linked lists of nodes at each depth
def getDepths(head):
    if head == None:
        return None
    visited = set()
    toVisit = Queue()
    toVisit.enqueue((head, 0))
    depths = []
    while len(toVisit) > 0:
        (visiting, depth) = toVisit.dequeue()
        if visiting not in visited:
            if visiting.left != None:
                toVisit.enqueue((visiting.left, depth + 1))
            if visiting.right != None:
                toVisit.enqueue((visiting.right, depth + 1))
        if depth == len(depths):
            depths.append(None)
        visiting.next = depths[depth]
        depths[depth] = visiting
        visited.add(visiting)
    return depths

# 4.4 Check Balanced: Check if a binary tree is balanced (heights of two subtrees of a node differ by at most 1)
def isBalanced(root):
    return isBalancedRec(root) is not False

def isBalancedRec(root):
    if root == None:
        return 0
    leftHeight = isBalancedRec(root.left)
    rightHeight = isBalancedRec(root.right)
    if leftHeight is False or rightHeight is False or abs(leftHeight - rightHeight) > 1:
        #print('Unbalanced subtrees {2} and {3} with heights {0} and {1}'.format(leftHeight, rightHeight, root.left, root.right))
        return False
    return 1 + max(leftHeight, rightHeight)

# 4.5 ValidateBST: Check if a binary tree is a binary search tree
def validateBST(root):
    return validateBSTRec(root, -maxsize - 1, maxsize)

def validateBSTRec(root, min, max):
    if root == None:
        return True
    return (min < root.data and 
            root.data < max and
            validateBSTRec(root.left, min, root.data) and
            validateBSTRec(root.right, root.data, max))

# 4.6 Sucessor: Find the inorder sucessor of a given node in a BST.  Nodes have a parent link
def getInorderSucessor(root):
    if root == None:
        return None
    if (root.right == None and
        root.parent != None and
        root.parent.data > root.data):
            return root.parent
    current = root.right
    while current.left != None:
        current = current.left
    return current

# 4.7 Build Order: Given a list of projects and a list of (dependecy, dependant), find a build
# order that will allow the projects to be built.  Return an error if it is not possible.
def getBuildOrder(projects, dependencies):
    if projects == None:
        return None
    if dependencies == None:
        return list(OrderedDict.fromkeys(projects))
    build = []
    nextLevel = projects[:]
    while len(nextLevel) > 0:
        #print('nextLevel: {0}'.format(nextLevel))
        build.append(nextLevel)
        
        currentLevel = build[-1]
        nextLevel = list(OrderedDict.fromkeys([dependant for (dependency, dependant) in dependencies if dependency in currentLevel]))
        if currentLevel == nextLevel:
            raise BuildDependencyCycleError(currentLevel)

        build[-1] = [project for project in currentLevel if project not in nextLevel]

    return reduce(lambda finalBuild, level: finalBuild + level, build)

class Error(Exception):
    pass

class BuildDependencyCycleError(Error):
    def __init__(self, message):
        self.message = message

# 4.8 First Common Ancestor: Find the first common ancestor of two binary nodes.  Avoid storing
# additional nodes in a data structure

def firstCommonAncestor(root, a, b):
    if root == None:
        return None
    
    aLeft = root.left != None and a in root.left
    aRight = root.right != None and a in root.right
    bLeft = root.left != None and b in root.left
    bRight = root.right != None and b in root.right

    if ((root.data == a and (bLeft or bRight)) or
        (root.data == b and (aLeft or aRight)) or
        (aLeft and bRight) or
        (bLeft and aRight)):
        return root

    if ((not aLeft) and (not aRight) or
        (not bLeft) and (not bRight)):
        return None

    return firstCommonAncestor(
        root.left if aLeft and bLeft else root.right,
        a,
        b)

# 4.9 BST Sequences: A BST was created by inserting elements from an array (in order).  Given a BST
# with distinct elements, print all possible arrays that could have led to this tree

def BSTSequenceMix(leftSource, rightSources):
    #print('left: {0}, rights: {1}'.format(leftSource, rightSources))
    if leftSource == None or len(leftSource) == 0:
        return rightSources
    sources = []
    for rightSource in rightSources:
        if len(rightSource) == 0:
            sources.append(leftSource)
            continue
        for i in range(0, len(leftSource) + 1):
            prefix = leftSource[0:i] + [rightSource[0]]
            #print('prefix: {0}'.format(prefix))
            suffices = BSTSequenceMix(leftSource[i:], [rightSource[1:]])
            #print('suffices: {0}'.format(suffices))

            for suffix in suffices:
                sources.append(prefix + suffix)

    return sources

def getSequenceArrays(root):
    if root == None:
        return [[]]

    leftSources = getSequenceArrays(root.left)
    rightSources = getSequenceArrays(root.right)
    #print('lefts: {0}, rights: {1}'.format(leftSources, rightSources))

    sources = []
    for leftSource in leftSources:
        for source in BSTSequenceMix(leftSource, rightSources):
            sources.append([root.data] + source)

    return sources

class Test(unittest.TestCase):

    def test_pathExists(self):
        b = GraphNode('!')
        a = GraphNode('h', [GraphNode('e', 
                                    [GraphNode('l'), GraphNode('l'), GraphNode('o', 
                                                                                    [b])])])
        self.assertTrue(pathExists(a, b))
        self.assertTrue(pathExists(b, a))

    def test_createBST(self):
        arr = [x for x in range(0, 11)]
        BST = createBST(arr)
        for i in range(0, 11):
            self.assertTrue(i in BST)
        self.assertTrue(-1 not in BST)
        self.assertTrue(11 not in BST)

    def test_getDepths(self):
        binaryTree = BinaryNode(1, 
                    BinaryNode(2,
                        BinaryNode(4)), 
                    BinaryNode(3,
                        BinaryNode(5,
                            BinaryNode(6),
                            BinaryNode(7,
                                None,
                                BinaryNode(8)))))
        depths = getDepths(binaryTree)
        self.assertIsNotNone(depths)
        self.assertTrue(
            depths[0].data == 1 and
            depths[0].next == None and
            depths[1].data == 3 and
            depths[1].next.data == 2 and
            depths[1].next.next == None and
            depths[2].data == 5 and
            depths[2].next.data == 4 and
            depths[2].next.next == None and
            depths[3].data == 7 and
            depths[3].next.data == 6 and
            depths[3].next.next == None and
            depths[4].data == 8 and
            depths[4].next == None and
            len(depths) == 5)

    def test_isBalanced(self):
        self.assertTrue(isBalanced(None))
        self.assertTrue(isBalanced(BinaryNode(1)))
        unBalancedBinaryTree = BinaryNode(1, 
                                BinaryNode(2,
                                    BinaryNode(4)), 
                                BinaryNode(3,
                                    BinaryNode(5,
                                        BinaryNode(6),
                                        BinaryNode(7,
                                            None,
                                            BinaryNode(8)))))
        self.assertFalse(isBalanced(unBalancedBinaryTree))
        balancedBinaryTree = BinaryNode(1, 
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
        self.assertTrue(isBalanced(balancedBinaryTree))

    def test_validateBST(self):
        balancedBinaryTree = BinaryNode(1, 
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
        binarySearchTree = BinarySearchNode(10, 
                                BinarySearchNode(4,
                                    BinarySearchNode(3)), 
                                BinarySearchNode(16,
                                    BinarySearchNode(12,
                                        BinarySearchNode(11),
                                        BinarySearchNode(13,
                                            None,
                                            BinarySearchNode(14)))))
        self.assertFalse(validateBST(balancedBinaryTree))
        self.assertTrue(validateBST(binarySearchTree))

    def test_getInorderSucessor(self):
        balancedBinarySearchTree = BinarySearchNode(1, 
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

        sucessor = getInorderSucessor(binarySearchTree)
        self.assertTrue(sucessor != None and sucessor.data == 11)
        sucessor = getInorderSucessor(sucessor)
        self.assertTrue(sucessor != None and sucessor.data == 12)
        sucessor = getInorderSucessor(sucessor)
        self.assertTrue(sucessor != None and sucessor.data == 13)
        sucessor = getInorderSucessor(sucessor)
        self.assertTrue(sucessor != None and sucessor.data == 14)

    def test_getBuildOrder(self):
        projects = ['a','b','c','d','e','f']
        dependencies = [('a','d'), ('f','b'), ('b','d'), ('f','a'), ('d','c')]
       
        self.assertIsNone(getBuildOrder(None, dependencies))
        
        build = getBuildOrder(projects, None)
        self.assertIsNotNone(build)
        self.assertEqual(len(build),len(set(build)))
        
        build = getBuildOrder(projects, dependencies)
        self.assertIsNotNone(build) 
        self.assertEqual(len(build), len(set(build)))
        self.assertTrue(all([build.index(dependency) < build.index(dependant) for (dependency,dependant) in dependencies]))
        
        dependencies = [('a','b'), ('b','c'), ('c', 'a')]
        self.assertRaises(BuildDependencyCycleError, getBuildOrder, projects, dependencies)

    def test_firstCommonAncestor(self):
        self.assertIsNone(firstCommonAncestor(None, 4, 1))
        self.assertIsNone(firstCommonAncestor(BinaryNode('a'), 'b', 'c'))
        balancedBinaryTree = BinaryNode(1, 
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
        ancestor = firstCommonAncestor(balancedBinaryTree, 3, 9)
        self.assertIsNotNone(ancestor)
        self.assertEqual(ancestor.data, 2)
        ancestor = firstCommonAncestor(balancedBinaryTree, 1, 8)
        self.assertIsNotNone(ancestor)
        self.assertEqual(ancestor.data, 1)
        ancestor = firstCommonAncestor(balancedBinaryTree, 4, 6)
        self.assertIsNotNone(ancestor)
        self.assertEqual(ancestor.data, 1)

    def test_getSequenceArrays(self):
        sequences = BSTSequenceMix([1], [[3]])
        self.assertEqual(len(sequences), 2)
        self.assertTrue([1, 3] in sequences and [3, 1] in sequences)
        sequences = BSTSequenceMix([2, 1, 3], [[5, 7]])
        sequences = getSequenceArrays(BinarySearchNode(2, BinarySearchNode(1), BinarySearchNode(3)))
        self.assertEqual(len(sequences), 2)
        self.assertTrue([2, 1, 3] in sequences and [2, 3, 1] in sequences)
        sequences = getSequenceArrays(
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
        sequences = getSequenceArrays(
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


if __name__ == '__main__':

    unittest.main()


    
    

    
    
