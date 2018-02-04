import unittest

class Node():
    def __init__(self, data, next=None):
        self.data = data
        self.next = next

    def __len__(self):
        if self.next == None:
            return 1
        else:
            return 1 + len(self.next)

# 2.1 Return kth to last: Return the kth last element of a singly linked list
def getKthLast(head, k):
    if k < 1:
        return None
    current = head
    while k > 0:
        if current == None:
            return None
        current = current.next
        k -= 1
    kthLast = head
    while current != None:
        current = current.next
        kthLast = kthLast.next
    return kthLast

# 2.3 Delete Middle Node: Given only a node, except the first/last, delete that node from a singly linked list
def deleteNode(toDelete):
    if toDelete == None:
        return
    toDelete.data = toDelete.next.data
    if toDelete.next.next == None:
        toDelete.next = None
        return
    deleteNode(toDelete.next)

# 2.4 Partition: Partition a linked list around a value x.  x can be anywhere in the right partition
def roughPartition(head, x):
    leftHead = leftTail = rightHead = prev = None
    current = head
    while current != None:
        if current.data < x:
            if prev != None:
                prev.next = current.next
            if leftHead == None:
                leftHead = current
            else:
                leftTail.next = current
            leftTail = current
        else:
            if rightHead == None:
                rightHead = current
            prev = current
        current = current.next

    if leftHead == None:
        return rightHead
    leftTail.next = rightHead
    return leftHead

# 2.5 Sum Lists: Given two numbers in reverse-digit list format, return the sum in the same format
def sumLists(operand1, operand2):
    sum = []
    head = tail = None
    larger = operand1 if len(operand1) >= len(operand2) else operand2
    smaller = operand2 if len(operand1) >= len(operand2) else operand1
    carry = 0
    while larger != None or carry > 0:
        largerData = 0 if larger == None else larger.data
        smallerData = 0 if smaller == None else smaller.data
        if largerData > 9 or largerData < -9 or smallerData > 9 or smallerData < -9:
            print('sumLists: Input list must contain nodes with single-digit data')
            return None

        digit = Node((largerData + smallerData + carry) % 10)
        carry = int((largerData + smallerData + carry) / 10)

        if head == None:
            head = digit
        else:
            tail.next = digit
        tail = digit

        if larger != None:
            larger = larger.next
        if smaller != None:
            smaller = smaller.next

    return head

# 2.6 Palindrome: Check if a linked list is a palindrome
def isPalindrome(head):
    length = 0
    current = head
    while current != None:
        current = current.next
        length += 1
    
    current = head
    half = []
    for i in range(0, int(length / 2)):
        half.append(current.data)
        current = current.next
    if length % 2 == 1:
        current = current.next
    for i in range(0, int(length / 2)):
        if current.data != half[len(half) - 1 - i]:
            #print('{0} != {1}'.format(current.data, half[len(half) - 1 - i]))
            return False

        #print('{0} == {1}'.format(current.data, half[len(half) - 1 - i]))
        current = current.next

    return True

# 2.7 Intersection: Find the intersecting node of two singly-linked lists
def getIntersection(head1, head2):
    visited = set()
    while head1 != None:
        visited.add(head1)
        head1 = head1.next
    while head2 != None:
        if head2 in visited:
            return head2
        head2 = head2.next
    return None

# 2.8 Loop Detection: Given a linked list containing a loop, return the node at the start of the loop
def getLoop(head):
    visited = set()
    while head != None:
        if head in visited:
            return head
        visited.add(head)
        head = head.next

# Utility functions

def isEquivalent(head1, head2):
    return (head1 == None and head2 == None) or (head1 != None and head2 != None and head1.data == head2.data and isEquivalent(head1.next, head2.next))

def printList(head):
    if head == None:
        print(None)
    else:
        print(head.data, end=' ')
        printList(head.next)


class Test(unittest.TestCase):
    def setUp(self):
        self.empty = None
        self.single = Node(1, None)
        self.canonical = Node(1, Node(2, Node(3, Node(4, Node(5, Node(6, Node(7, Node(8, Node(9, Node(10))))))))))
        self.canonical2 = Node(1, Node(2, Node(3, Node(4, Node(5, Node(6, Node(7, Node(8, Node(9)))))))))
        self.canonical3 = Node(7, Node(8, Node(9)))

    def test_getKthLast(self):
        self.assertIsNone(getKthLast(self.empty, 3))
        self.assertIsNone(getKthLast(self.single, 2))
        self.assertEqual(getKthLast(self.single, 1).data, 1)

    def test_deleteNode(self):
        deleteNode(self.canonical.next)
        self.assertTrue(isEquivalent(
                            self.canonical, 
                            Node(1, Node(3, Node(4, Node(5, Node(6, Node(7, Node(8, Node(9, Node(10)))))))))))

    def test_partitionTest(self):
        partitionSample = Node(3, Node(5, Node(8, Node(5, Node(10, Node(2, Node(1, None)))))))
        x = 5
        head = roughPartition(partitionSample, x)
        current = head
        while current.data != x:
            if current == x:
                break
            self.assertTrue(current.data < x, '{0} {1} found before {2}'.format(False, current.data, x))
            current = current.next
        while current != None:
            self.assertTrue(current.data >= x, '{0} {1} found after {2}'.format(False, current.data, x))
            current = current.next

    def test_sumLists(self):
        self.assertTrue(isEquivalent(
                            sumLists(self.canonical2, self.canonical3), 
                            Node(8, Node(0, Node(3, Node(5, Node(5, Node (6, Node(7, Node(8, Node(9)))))))))))

    def test_isPalindrome(self):
        self.assertFalse(isPalindrome(self.canonical))
        self.assertTrue(isPalindrome(Node(1, Node(2, Node(3, Node(4, Node(3, Node(2, Node(1)))))))))

    def test_getIntersection(self):
        self.assertIsNone(getIntersection(self.canonical, self.single))
        intersectingList = Node(1, Node(2, Node(3, Node(4))))
        intersectingList.next.next.next.next = self.canonical.next.next.next.next
        intersection = getIntersection(self.canonical, intersectingList)
        self.assertIsNotNone(intersection)
        self.assertEqual(intersection.data, 5)

    def test_getLoop(self):
        self.assertIsNone(getLoop(self.empty))
        self.assertIsNone(getLoop(self.canonical))
        tail = Node(10)
        loopedList = Node(1, Node(2, Node(3, Node(4, Node(5, Node(6, Node(7, Node(8, Node(9, tail)))))))))
        tail.next = loopedList.next.next.next.next
        loop = getLoop(loopedList)
        self.assertIsNotNone(loop)
        self.assertEqual(loop.data, 5)

if __name__ == '__main__':
    unittest.main()
    
    
    

    

    
