import unittest


class Node:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next

    def __len__(self):
        if self.next == None:
            return 1
        else:
            return 1 + len(self.next)


# 2.1 Return kth to last: Return the kth last element of a singly linked list
def get_kth_last(head, k):
    if k < 1:
        return None
    current = head
    while k > 0:
        if current is None:
            return None
        current = current.next
        k -= 1
    kth_last = head
    while current is not None:
        current = current.next
        kth_last = kth_last.next
    return kth_last


# 2.3 Delete Middle Node: Given only a node, except the first/last, delete that node from a singly linked list
def delete_node(to_delete):
    if to_delete is None:
        return
    to_delete.data = to_delete.next.data
    if to_delete.next.next is None:
        to_delete.next = None
        return
    delete_node(to_delete.next)


# 2.4 Partition: Partition a linked list around a value x.  x can be anywhere in the right partition
def rough_partition(head, x):
    left_head = left_tail = right_head = prev = None
    current = head
    while current is not None:
        if current.data < x:
            if prev is not None:
                prev.next = current.next
            if left_head is None:
                left_head = current
            else:
                left_tail.next = current
            left_tail = current
        else:
            if right_head is None:
                right_head = current
            prev = current
        current = current.next

    if left_head is None:
        return right_head
    left_tail.next = right_head
    return left_head


# 2.5 Sum Lists: Given two numbers in reverse-digit list format, return the sum in the same format
def sum_lists(operand1, operand2):
    head = tail = None
    larger = operand1 if len(operand1) >= len(operand2) else operand2
    smaller = operand2 if len(operand1) >= len(operand2) else operand1
    carry = 0
    while larger is not None or carry > 0:
        larger_data = 0 if larger is None else larger.data
        smaller_data = 0 if smaller is None else smaller.data
        if larger_data > 9 or larger_data < -9 or smaller_data > 9 or smaller_data < -9:
            print('sumLists: Input list must contain nodes with single-digit data')
            return None

        digit = Node((larger_data + smaller_data + carry) % 10)
        carry = int((larger_data + smaller_data + carry) / 10)

        if head is None:
            head = digit
        else:
            tail.next = digit
        tail = digit

        if larger is not None:
            larger = larger.next
        if smaller is not None:
            smaller = smaller.next

    return head


# 2.6 Palindrome: Check if a linked list is a palindrome
def is_palindrome(head):
    length = 0
    current = head
    while current is not None:
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
def get_intersection(head1, head2):
    visited = set()
    while head1 is not None:
        visited.add(head1)
        head1 = head1.next
    while head2 is not None:
        if head2 in visited:
            return head2
        head2 = head2.next
    return None


# 2.8 Loop Detection: Given a linked list containing a loop, return the node at the start of the loop
def get_loop(head):
    visited = set()
    while head is not None:
        if head in visited:
            return head
        visited.add(head)
        head = head.next

# Utility functions




class Test(unittest.TestCase):
    def is_equivalent(self, head1, head2):
        return (head1 is None and head2 is None)\
               or (head1 is not None
                   and head2 is not None
                   and head1.data == head2.data
                   and self.is_equivalent(head1.next,head2.next))

    def setUp(self):
        self.empty = None
        self.single = Node(1, None)
        self.canonical = Node(1, Node(2, Node(3, Node(4, Node(5, Node(6, Node(7, Node(8, Node(9, Node(10))))))))))
        self.canonical2 = Node(1, Node(2, Node(3, Node(4, Node(5, Node(6, Node(7, Node(8, Node(9)))))))))
        self.canonical3 = Node(7, Node(8, Node(9)))

    def test_getKthLast(self):
        self.assertIsNone(get_kth_last(self.empty, 3))
        self.assertIsNone(get_kth_last(self.single, 2))
        self.assertEqual(get_kth_last(self.single, 1).data, 1)

    def test_deleteNode(self):
        delete_node(self.canonical.next)
        self.assertTrue(self.is_equivalent(
                            self.canonical, 
                            Node(1, Node(3, Node(4, Node(5, Node(6, Node(7, Node(8, Node(9, Node(10)))))))))))

    def test_partitionTest(self):
        partition_sample = Node(3, Node(5, Node(8, Node(5, Node(10, Node(2, Node(1, None)))))))
        x = 5
        head = rough_partition(partition_sample, x)
        current = head
        while current.data != x:
            if current == x:
                break
            self.assertTrue(current.data < x, '{0} {1} found before {2}'.format(False, current.data, x))
            current = current.next
        while current is not None:
            self.assertTrue(current.data >= x, '{0} {1} found after {2}'.format(False, current.data, x))
            current = current.next

    def test_sumLists(self):
        self.assertTrue(self.is_equivalent(
                            sum_lists(self.canonical2, self.canonical3),
                            Node(8, Node(0, Node(3, Node(5, Node(5, Node (6, Node(7, Node(8, Node(9)))))))))))

    def test_isPalindrome(self):
        self.assertFalse(is_palindrome(self.canonical))
        self.assertTrue(is_palindrome(Node(1, Node(2, Node(3, Node(4, Node(3, Node(2, Node(1)))))))))

    def test_getIntersection(self):
        self.assertIsNone(get_intersection(self.canonical, self.single))
        intersecting_list = Node(1, Node(2, Node(3, Node(4))))
        intersecting_list.next.next.next.next = self.canonical.next.next.next.next
        intersection = get_intersection(self.canonical, intersecting_list)
        self.assertIsNotNone(intersection)
        self.assertEqual(intersection.data, 5)

    def test_getLoop(self):
        self.assertIsNone(get_loop(self.empty))
        self.assertIsNone(get_loop(self.canonical))
        tail = Node(10)
        looped_list = Node(1, Node(2, Node(3, Node(4, Node(5, Node(6, Node(7, Node(8, Node(9, tail)))))))))
        tail.next = looped_list.next.next.next.next
        loop = get_loop(looped_list)
        self.assertIsNotNone(loop)
        self.assertEqual(loop.data, 5)


if __name__ == '__main__':
    unittest.main()
    

