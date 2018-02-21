from functools import reduce
import unittest


# 3.3 Stack of Plates: Define a data structure which implements Stack (push/pop) using a collection of
# maximally sized stacks.  Also implement popAt for popping a particular stack.
class SetofStacks:
    def __init__(self, threshold):
        self.threshold = threshold
        self.stacks = [[]]

    def push(self, el):
        if len(self.stacks[0]) >= self.threshold:
            self.stacks = [[]] + self.stacks
        self.stacks[0] = [el] + self.stacks[0]

    def pop(self):
        if len(self.stacks[0]) > 0:
            el = self.stacks[0][0]
            del self.stacks[0][0]
            if len(self.stacks) > 1 and len(self.stacks[0]) < 1:
                del self.stacks[0]
            return el

    def pop_at(self, i):
        if i >= 0 and i < len(self.stacks) and len(self.stacks[i]) > 0:
            el = self.stacks[i][0]
            del self.stacks[i][0]
            if len(self.stacks[i]) < 1 and len(self.stacks) > 1:
                del self.stacks[i]
            return el

    def __str__(self):
        return str([str(stack) + ' ' for stack in self.stacks])

    def __len__(self):
        return reduce(lambda x,y: x+y, [len(stack) for stack in self.stacks])


# 3.4 Queue via Stacks: Implement a Queue class using two stacks
class Queue:
    def __init__(self):
        self.entrance = SetofStacks(10)
        self.exit = SetofStacks(10)

    def enqueue(self, el):
        self.entrance.push(el)

    def dequeue(self):
        if len(self.exit) < 1:
            while len(self.entrance) > 0:
                self.exit.push(self.entrance.pop())
        return self.exit.pop()

# 3.5 Sort Stack: Sort a stack, smallest on top, using only 1 extra temp stack with push/pop/peek/isEmpty
def sort_stack(stack):
    buffer = SetofStacks(10)
    unsorted_length = len(stack)
    while unsorted_length > 0:
        for i in range(0, unsorted_length):
            maxElement = None
            el = stack.pop()
            if maxElement != None and el < maxElement:
                buffer.push(el)
            else:
                buffer.push(maxElement)
                maxElement = el
            stack.push(maxElement)
            unsorted_length -= 1
        for i in range(0, unsorted_length):
            stack.push(buffer.pop())


# 3.6 Animal Shelter
class Node:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next

    def __len__(self):
        if self.next is None:
            return 1
        else:
            return 1 + len(self.next)

    def __str__(self):
        return 'Node:' + str(self.data)


class LinkedList:
    def __init__(self):
        self.head = None
        self.length = 0

    # Wraps given element in Node and then pushes
    def push_head(self, el):
        head = Node(el)
        head.next = self.head
        self.head = head
        self.length += 1

    # Returns head value, unwrapped from the Node
    def pop_head(self):
        head = self.head
        self.length = self.length if head is None else self.length - 1
        self.head = None if self.head is None else self.head.next
        return head.data

    def __str__(self):
        if self.head is None:
            return '[]'
        else:
            return '[{0}]'.format(self.stringify(self.head))

    def stringify(self, head):
        if head is None:
            return str(None)
        else:
            return str(head.data) + ', ' + self.stringify(head.next)

    def __len__(self):
        return self.length


class PriorityQueue:
    def __init__(self):
        self.entrance = LinkedList()
        self.exit = LinkedList()

    def enqueue(self, el, priority):
        self.entrance.push_head((el, priority))

    def check_and_fill_exit(self):
        if len(self.exit) < 1:
            while len(self.entrance) > 0:
                self.exit.push_head(self.entrance.pop_head())

    def dequeue(self):
        self.check_and_fill_exit()
        head = self.exit.pop_head()
        return head if head is not None else None

    def peek(self):
        self.check_and_fill_exit()
        head = self.exit.head
        head = head.data if head is not None else None
        return head


class Dog:
    def __init__(self):
        None

    def __str__(self):
        return 'Dog'

    def __repr__(self):
        return 'Dog()'


class Cat:
    def __init__(self):
        None

    def __str__(self):
        return 'Cat'

    def __repr__(self):
        return 'Cat()'


class ShelterQueue:
    def __init__(self):
        self.priority = 0
        self.dogQueue = PriorityQueue()
        self.catQueue = PriorityQueue()

    def enqueue(self, animal):
        if type(animal) is Dog:
            self.dogQueue.enqueue(animal, self.priority)
        elif type(animal) is Cat:
            self.catQueue.enqueue(animal, self.priority)
        else:
            return
        self.priority += 1

    def dequeue_dog(self):
        return self.dogQueue.dequeue()[0] if self.dogQueue.peek() is not None else None

    def dequeue_cat(self):
        return self.catQueue.dequeue()[0] if self.catQueue.peek() is not None else None

    def dequeueAny(self):
        first_dog = self.dogQueue.peek()
        first_cat = self.catQueue.peek()
        (dog, dogPriority) = first_dog if first_dog is not None else (None, None)
        (cat, catPriority) = first_cat if first_cat is not None else (None, None)
        if catPriority is None or (dogPriority is not None and dogPriority < catPriority):
            (dog, dogPriority) = self.dogQueue.dequeue() if first_dog is not None else (None, None)
            return dog
        else:
            (cat, catPriority) = self.catQueue.dequeue() if first_cat is not None else (None, None)
            return cat


class Test(unittest.TestCase):
    def test_SetofStacks(self):
        stack = SetofStacks(5)

        contents = []
        for i in range(10, 0, -1):
            stack.push(i)
        for i in range(1, 11):
            el = stack.pop()
            if el is not None:
                contents.append(el)
        self.assertEqual(contents, [i for i in range(1, 11)])

        contents = []
        for i in range(20, 0, -1):
            stack.push(i)
        for i in range(0, 21):
            el = stack.pop_at(0)
            if el is not None:
                contents.append(el)
        self.assertEqual(contents, [i for i in range(1, 21)])

        contents = []
        for i in range(20, 0, -1):
            stack.push(i)
        for i in range(0, 6):
            el = stack.pop_at(2)
            if el is not None:
                contents.append(el)
        self.assertEqual(contents, [11, 12, 13, 14, 15, 16])
        contents = []
        for i in range(0, 15):
            el = stack.pop_at(0)
            if el is not None:
                contents.append(el)
        self.assertEqual(contents, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20])

    def test_Queue(self):
        queue = Queue()
        contents = []
        for i in range(1, 11):
            queue.enqueue(i)
        for i in range(1, 11):
            el = queue.dequeue()
            if el is not None:
                contents.append(el)
        for i in range(11, 21):
            queue.enqueue(i)
        for i in range(11, 21):
            el = queue.dequeue()
            if el is not None:
                contents.append(el)
        self.assertEqual(contents, [i for i in range(1, 21)])

        unsorted_stack = SetofStacks(5)
        contents = []
        for i in range(20, 0, -1):
            unsorted_stack.push(i)
        sort_stack(unsorted_stack)
        for i in range(20, 0, -1):
            el = unsorted_stack.pop()
            if el is not None:
                contents.append(el)
        self.assertEqual(contents, [i for i in range(1, 21)])

    def test_ShelterQueue(self):
        shelter = ShelterQueue()
        shelter.enqueue(Cat())
        shelter.enqueue(Cat()) 
        shelter.enqueue(Dog()) 
        shelter.enqueue(Cat())
        shelter.enqueue(Cat())
        shelter.enqueue(Dog())
        shelter.enqueue(Cat())
        shelter.enqueue(Cat())
        self.assertTrue(type(shelter.dequeueAny()) is Cat and
                        type(shelter.dequeue_dog()) is Dog and
                        type(shelter.dequeueAny()) is Cat and
                        type(shelter.dequeue_cat()) is Cat and
                        type(shelter.dequeueAny()) is Cat and
                        type(shelter.dequeueAny()) is Dog and
                        type(shelter.dequeueAny()) is Cat and
                        type(shelter.dequeueAny()) is Cat)


if __name__ == '__main__':
    unittest.main()

