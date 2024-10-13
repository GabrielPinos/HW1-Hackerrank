# Say "Hello, World!" With Python
a = 'Hello, World!'
if a == 'Hello, World!':
    print(a)

# Python If-Else
#!/bin/python3
import math
import os
import random
import re
import sys
def control_number(n):
    if n % 2 != 0:  
        print('Weird')
    elif 2 <= n <= 5:  
        print('Not Weird')
    elif 6 <= n <= 20:  
        print('Weird')
    else: 
        print('Not Weird')
if __name__ == '__main__':
    while True:
        n = int(input('').strip())
        
        if n > 100 or n < 1:
            print("Please insert an integer between 1 and 100")
        else:
            break 
    control_number(n)

# Arithmetic Operators

def sum_diff_prod(a,b):
    summ = a + b
    diff  = a - b
    prod = a * b
    
    print(summ)
    print(diff)
    print(prod)
    
    
    
if __name__ == '__main__':    
    while True:
            a = int(input())
            b = int(input())
            
            if a > 10**10 or a < 1:
                print("Please insert an integer between 1 and 100")
            elif b > 10**10 or b < 1:
                print("Please insert an integer between 1 and 100")
            else:
                break

    sum_diff_prod(a,b)


# Python: Division

def division(a,b):
    integer_div = a//b
    float_div = a/b
    
    print(integer_div)
    print(float_div)
    
    
    
if __name__ == '__main__':    
    while True:
            a = int(input())
            b = int(input())
            
            if a > 10**10 or a < 1:
                print("Please insert an integer between 1 and 100")
            elif b > 10**10 or b < 1:
                print("Please insert an integer between 1 and 100")
            else:
                break

    division(a,b)

# List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
coordinates = [
    [i, j, k]
    for i in range(x + 1)
    for j in range(y + 1)
    for k in range(z + 1)
    if (i + j + k) != n
]
# Print the resulting list
print(coordinates)
    

# Find the Runner-Up Score!

def runner_up(n, arr):
    first = -float('inf')
    second = -float('inf')
    
    for i in range(n):
        if arr[i] > first:
            second = first
            first = arr[i]
        elif arr[i] > second and arr[i] != first:
            second = arr[i]
    
    print(second)

if __name__ == '__main__':
    n = int(input())
    if n < 2 or n > 10:
        print("Error")
    else:
        arr = list(map(int, input().split()))
        
        
        if all(-100 <= x <= 100 for x in arr):
            runner_up(n, arr)
        else:
            print("Error")

# Nested Lists
if __name__ == '__main__':
    students=[]
    N=int(input())
    if N < 2 or N > 5:
        print("Error")
    else :
        for _ in range(N):
            name = input()
            score = float(input())
            students.append([name, score])


    scores = sorted(set([student[1] for student in students]))
    second_lowest_score = scores[1]
    second_lowest_students = [student[0] for student in students if student[1] == second_lowest_score]
    second_lowest_students.sort()
    for name in second_lowest_students:
        print(name)


# Finding the percentage
if __name__ == '__main__':
    n = int(input())
    if n < 2 or n > 10:
        print("Error")
    else:
        student_marks = {}
        for _ in range(n):
            name, *line = input().split()
            scores = list(map(float, line))
            if len(scores) != 3:
                print("Error")
                break
            if not all(0 <= score <= 100 for score in scores):
                print("Error")
                break
            
            student_marks[name] = scores  
        query_name = input()
        
for i in range (n):
        if query_name in student_marks:
            scores = student_marks[query_name]
            mean = sum(scores) / len(scores)
print(f"{mean:.2f}")
        

# Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
   
t=tuple(integer_list) 
    
print(hash(t))

# Lists
if __name__ == '__main__':
    N = int(input())
    
    
a=[]
for _ in range(N):
    command = input().split()
    
    if command[0] == "insert":
        a.insert(int(command[1]), int(command[2]))
    elif command[0] == "print":
        print(a)
    elif command[0] == "remove":
        a.remove(int(command[1]))
    elif command[0] == "append":
        a.append(int(command[1]))
    elif command[0] == "sort":
        a.sort()
    elif command[0] == "pop":
        a.pop()
    elif command[0] == "reverse":
        a.reverse()

# sWAP cASE
def swap_case(s):
    string = s.swapcase()
    return string

# String Split and Join

def split_and_join(line):
    # write your code here
    line = line.split(" ") 
    line="-".join(line)
    return line
if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

# What's Your Name?
#
# Complete the 'print_full_name' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. STRING first
#  2. STRING last
#
def print_full_name(first, last):
    if len(first) > 10 :
        print(Error)
    elif len(last) > 10 :
        print(Error)
    else :
        a = str("Hello "+ (first) +" " + (last) + "! You just delved into python.")
    print (a)

# Mutations
def mutate_string(string, position, character):
    mutate = list(string)
    mutate[position]= character
    n_string="".join(mutate)
    return n_string

# Find a string
def count_substring(string, sub_string):
    if len(string)<1 or len(string)>200:
        print("Error")
    else:
        count= 0
        for i in range (0, len(string)- len(sub_string) + 1):
            if string[i:i + len(sub_string)]== sub_string:
              count+=1   
    return count

# String Validators
if __name__ == '__main__':
    s = input()
    print(any(c.isalnum() for c in s))
    
    print(any(c.isalpha() for c in s))
    
    print(any(c.isdigit() for c in s))
    
    print(any(c.islower() for c in s))
    
    print(any(c.isupper() for c in s))

# Text Wrap

def wrap(string, max_width):
    wrapped_text = textwrap.fill(string, max_width)
    return wrapped_text


# Designer Door Mat
# Enter your code here. Read input from STDIN. Print output to STDOUT
def print_door_mat(N, M):
    for i in range(1, N, 2): 
        print((".|." * i).center(M, "-"))
    print("WELCOME".center(M, "-"))
    
    for i in range(N-2, -1, -2):
        print((".|." * i).center(M, "-"))
if __name__ == '__main__':
    N, M = map(int, input().split())
    if 5 < N < 101 and 15 < M < 303 and M == 3 * N:
        print_door_mat(N, M)

# String Formatting
def print_formatted(number):
    width = len(bin(number)) - 2 
    if number>=1 and number<=99:
        for i in range(1, number + 1):
            print(f"{str(i).rjust(width)} {oct(i)[2:].rjust(width)} {hex(i)[2:].upper().rjust(width)} {bin(i)[2:].rjust(width)}")
    else:
        print ("Error")


# Text Alignment

if __name__ == '__main__':
  n = int(input())

h= 'H'
for i in range(n):
    print((h*i).rjust(n-1)+h+(h*i).ljust(n-1))
for i in range(n+1):  print((h*n).center(n*2)+(h*n).center(n*6))
for i in range((n+1)//2):
    print((h*n*5).center(n*6))    
for i in range(n+1): print((h*n).center(n*2)+(h*n).center(n*6))    
for i in range(n):  print(((h*(n-i-1)).rjust(n)+h+(h*(n-i-1)).ljust(n)).rjust(n*6))


# Alphabet Rangoli

alpha = "abcdefghijklmnopqrstuvwxyz"
def print_rangoli(size):
    row = []
    for i in range(size):
        stamp = "-".join(alpha[i:size])
        row.append(stamp[::-1] + stamp[1:])
    width = len(row[0])
    
    for i in range(size-1, 0, -1):
        print(row[i].center(width, '-'))
        
    for i in range(size):
        print(row[i].center(width, '-'))


# Capitalize!

def solve(s):
    return ' '.join(word.capitalize() for word in s.split(' '))


# The Minion Game
v='AEIOU'
   
def minion_game(string):
    stuart_score = 0
    kevin_score = 0
    length = len(string)
    for i in range(length):
        if string[i] in v:
            kevin_score += (length - i)
        else:
            stuart_score += (length - i)
    
    if stuart_score > kevin_score:
        print("Stuart", stuart_score)
    elif kevin_score > stuart_score:
        print("Kevin", kevin_score)
    else:
        print("Draw")


# Merge the Tools!
def merge_the_tools(string, k):
    split= (len(string))//k
    for i in range(0, len(string), k):
        block= string[i:i+k]
        char= []
        for c in block:
            if c not in char:
                char.append(c)
        print(''.join(char))


# Introduction to Sets
def average(array):
    if n<0 or n>100:
        return 'Error' 
    else :
        set1 = set(array)
        avg = sum(set1)/len(set1)
        return avg
    
    # your code goes here

# Symmetric Difference
# Enter your code here. Read input from STDIN. Print output to STDOUT
def simm_diff(a,b):
    if len(a)!= N:
        return "Error"
    if len(b)!= M:
        return "Error"
    else:
        a = set(a)
        b = set(b)
        result = a.symmetric_difference(b)
        return sorted(result)

if __name__ == '__main__':
    N = int(input())
    myset1 = list(map(int, input().split()))
    M=  int(input())
    myset2 = list(map(int, input().split()))
    result = simm_diff(myset1,myset2)
    for num in result:
        print(num)

# Set .add()
# Enter your code here. Read input from STDIN. Print output to STDOUT

if __name__ == '__main__':
    N = int(input())
    country = set()
    for i in range (N) :
        country.add(input())
    print (len(country))

# No Idea!
if __name__ == '__main__':
    n, m = map(int, input().split())
    if not (1 <= n <= 10**5) or not (1 <= m <= 10**5):
        print("Error")
        exit()
    array = list(map(int, input().split()))
    if len(array) != n or not all(1 <= num <= 10**9 for num in array):
        print("Error")
        exit()
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    if len(A) != m or len(B) != m:
        print("Error")
        exit()
    happiness = 0
    A= set(A)
    B = set(B)
    for num in array :
        if num in A:
            happiness += 1
        elif num in B:
            happiness -= 1
    print(happiness)


# Set .discard(), .remove() & .pop()
if __name__ == '__main__':
    n = int(input())
    myset = list(map(int, input().split()))
    if len(myset) != n:
        print("Error")
    else:
        for num in myset:
            if num < 0 or num > 9:
                print("Error")
                break
        else:
            myset = set(myset)
            num_commands = int(input())
            for _ in range(num_commands):
                command = input().split()
                if command[0] == "pop":
                    if myset:
                        myset.pop()
                elif command[0] == "remove":
                    try:
                        myset.remove(int(command[1]))
                    except KeyError:
                        pass
                elif command[0] == "discard":
                    myset.discard(int(command[1]))
               
            print(sum(myset))

# Set .union() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
if __name__ == '__main__':
    n = int, input().split()
    rolln = set(map(int, input().split()))
    b = int, input().split()
    rollb = set(map(int, input().split()))
    
    students = rolln.union(rollb)
    if len(students)> 1000 or len(students)<0:
        print("Error")
        exit()
    else:
        print(len(students))

# Set .intersection() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
# Enter your code here. Read input from STDIN. Print output to STDOUT
if __name__ == '__main__':
    n = int, input().split()
    rolln = set(map(int, input().split()))
    b = int, input().split()
    rollb = set(map(int, input().split()))
    
    students = rolln.intersection(rollb)
    if len(students)> 1000 or len(students)<0:
        print("Error")
        exit()
    else:
        print(len(students))

# Set .difference() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
if __name__ == '__main__':
    n = int, input().split()
    rolln = set(map(int, input().split()))
    b = int, input().split()
    rollb = set(map(int, input().split()))
    
    students = rolln.difference(rollb)
    if len(students)> 1000 or len(students)<0:
        print("Error")
        exit()
    else:
        print(len(students))

# Set .symmetric_difference() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
if __name__ == '__main__':
    n = int, input().split()
    rolln = set(map(int, input().split()))
    b = int, input().split()
    rollb = set(map(int, input().split()))
    
    students = rolln.symmetric_difference(rollb)
    if len(students)> 1000 or len(students)<0:
        print("Error")
        exit()
    else:
        print(len(students))

# Set Mutations
# Enter your code here. Read input from STDIN. Print output to STDOUT
# Enter your code here. Read input from STDIN. Print output to STDOUT
if __name__ == '__main__':
    n = int(input())
    A = set(map(int, input().split()))
    N = int(input())
    for _ in range(N):
        command, sets = input().split(), set(map(int, input().split()))
        if command[0] == "intersection_update":
            A.intersection_update(sets)
        elif command[0] == "update":
            A.update(sets)
        elif command[0] == "symmetric_difference_update":
            A.symmetric_difference_update(sets)
        elif command[0] == "difference_update":
            A.difference_update(sets)
    print(sum(A))

# The Captain's Room
# Enter your code here. Read input from STDIN. Print output to STDOUT
if __name__ == '__main__':
    K = int(input())
    numbers = list(map(int, input().split()))
    unique_numbers = set(numbers)
    captain_room = (sum(unique_numbers) * K - sum(numbers)) // (K - 1)
    print(captain_room)
    

# Check Subset
# Enter your code here. Read input from STDIN. Print output to STDOUT
if __name__ == '__main__':
    T = int(input())
    for _ in range(T): 
        nA, A = int(input()), set(map(int, input().split()))
        nB, B = int(input()), set(map(int, input().split()))
        print(A.issubset(B))


    

# Check Strict Superset
# Enter your code here. Read input from STDIN. Print output to STDOUT
if __name__ == '__main__':
    A = set(map(int, input().split()))
    n = int(input())
    boolean = False
    for _ in range(n):
        subset = set(map(int, input().split()))
        if subset.issubset(A) and len(A)>len(subset):
            boolean = True
        else:
           boolean = False
           break
    print(boolean)

# collections.Counter()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import Counter
if __name__ == '__main__':
    X = int(input())
    sizes = list(map(int, input().split()))
    N = int(input())
    wharehouse= Counter(sizes)
    amount = 0
    for i in range(N):
        shoe_size_xi  = list(map(int,input().split()))
        if wharehouse[shoe_size_xi[0]]>0:
            wharehouse[shoe_size_xi[0]] -=1
            amount += shoe_size_xi[1]
    print(amount)

# DefaultDict Tutorial
from collections import defaultdict
if __name__ == '__main__':
    n, m = map(int, input().split())
    A = defaultdict(list)
    for i in range(n):
        word = input()
        A[word].append(i + 1) 
    for _ in range(m):
        word = input()
        if word in A:
            print(" ".join(map(str, A[word])))
        else:
            print(-1)


# Collections.namedtuple()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import namedtuple
if __name__ == '__main__':
    N = int(input())
    columns = input().split()
    students = namedtuple("students", columns)
    marks = 0
    for _ in range (N):
        student_data= input().split()
        data = students(*student_data)
        marks += int(data.MARKS)
    average = marks/N
    print ( average)

# Collections.OrderedDict()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import OrderedDict
if __name__ == '__main__':
    N = int(input())
    ordered_dictionary = OrderedDict()
    
    for _ in range(N):
        item, price = input().rsplit(' ', 1)
        price = int(price)
        if item in ordered_dictionary:
            ordered_dictionary[item] += price
        else:
            ordered_dictionary[item] = price
    
    for item, net_price in ordered_dictionary.items():
        print(item, net_price)

# Word Order
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import Counter
if __name__ == '__main__':
    n = int(input())
    words = []
    for i in range(n):
        words.append(input().strip())
    word_count = Counter(words)
    print(len(word_count))
    print(' '.join(str(word_count[word]) for word in word_count))

# Collections.deque()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import deque
if __name__ == '__main__':
    d = deque()
    N = int(input())
    for i in range(N):
        command = input().split()
        if command[0] == "append":
            d.append(int(command[1]))
        elif command[0] == "appendleft":
            d.appendleft(int(command[1]))
        elif command[0] == "pop":
            d.pop()
        elif command[0] == "popleft":
            d.popleft()
    print(' '.join(map(str, d)))

# Piling Up!
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import deque
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        n = int(input())
        side_lengths = list(map(int, input().split()))
        d = deque(side_lengths)
        possible = True
        previous = float('inf')
        while d:
            if d[0] >= d[-1]:
                current = d.popleft()
            else:
                current = d.pop()
            if current > previous:
                possible = False
                break
            previous = current
        print("Yes" if possible else "No")

# Company Logo
#!/bin/python3
from collections import Counter
if __name__ == '__main__':
    s = input().strip()
    occ = Counter(s)
    # Get the three most common characters, sorted by count first, then alphabetically for ties
    most_common = occ.most_common()
    most_common.sort(key=lambda x: (-x[1], x[0]))
    
    for i in range(min(3, len(most_common))):
        print(most_common[i][0], most_common[i][1])

# Calendar Module
# Enter your code here. Read input from STDIN. Print output to STDOUT
import calendar
if __name__ == '__main__':
    month, day, year = map(int, input().split())
    n = calendar.weekday(year, month, day)
    name = calendar.day_name[n].upper()
    print(name)

# Time Delta
#!/bin/python3
import math
import os
import random
import re
import sys
from datetime import datetime

def time_delta(t1, t2):
    time_format = "%a %d %b %Y %H:%M:%S %z"
    
    time1 = datetime.strptime(t1, time_format)
    time2 = datetime.strptime(t2, time_format)
    
    if time1.year > 3000 or time2.year > 3000:
        raise ValueError
        
    delta = abs((time1 - time2).total_seconds())
    return str(int(delta))
if __name__ == '__main__':
    t = int(input())
    for t_itr in range(t):
        t1 = input().strip()
        t2 = input().strip()
        delta = time_delta(t1, t2)
        print(delta)

# Exceptions
# Enter your code here. Read input from STDIN. Print output to STDOUT

if __name__ == '__main__':
    T = int(input())

    for _ in range(T):
        a,b = input().split()
        try:
            print (int(a) // int(b))
        except ZeroDivisionError as e:
            print("Error Code:",e )
        except  ValueError as v:
            print("Error Code:",v )

# Zipped!
# Enter your code here. Read input from STDIN. Print output to STDOUT
if __name__ == '__main__':
    N, X = map(int, input().split())
    subjects = []
    for _ in range(X):
        subject = list(map(float, input().split()))
        subjects.append(subject)
    votes = zip(*subjects)
    for vote in votes:
        print(sum(vote) / X)

# Athlete Sort
#!/bin/python3
import math
import os
import random
import re
import sys
if __name__ == '__main__':
    nm = input().split()
    n = int(nm[0])
    m = int(nm[1])
    arr = []
    for _ in range(n):
        arr.append(list(map(int, input().split())))
    k = int(input())
    arr.sort(key=lambda x: x[k])
    for row in arr:
        print(' '.join(map(str, row)))

# ginortS
# Enter your code here. Read input from STDIN. Print output to STDOUT
def sort_string(s):
    lower = sorted([ch for ch in s if ch.islower()])
    upper = sorted([ch for ch in s if ch.isupper()])
    odd_digits = sorted([ch for ch in s if ch.isdigit() and int(ch) % 2 == 1])
    even_digits = sorted([ch for ch in s if ch.isdigit() and int(ch) % 2 == 0])
    
    result = ''.join(lower + upper + odd_digits + even_digits)
    return result
if __name__ == '__main__':
    s = input()
    print(sort_string(s))



# Map and Lambda Function
cube = lambda x: x**3
def fibonacci(n):
    a=[0,1]
    if n==0:
        return []
    elif n==1:
        return [0]
    elif n>2:
        for i in range(2, n):
            a.append(a[i-1]+a[i-2])
    return a[:n]

# XML 1 - Find the Score


def get_attr_number(node):
    attr_count = len(node.attrib)
    
    for child in node:
        attr_count += get_attr_number(child)
    
    return attr_count

# XML2 - Find the Maximum Depth

maxdepth = -1
def depth(elem, level):
    global maxdepth
    if (level == maxdepth):
        maxdepth += 1
    for a in elem:
        depth(a, level + 1)

# Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        formatted_numbers = ['+91 {} {}'.format(num[-10:-5], num[-5:]) for num in l]
        f(formatted_numbers)
    return fun

# Decorators 2 - Name Directory

from operator import itemgetter

def person_lister(f):
    def inner(people):
        people.sort(key=lambda x: int(x[2]))
        return [f(person) for person in people]
    return inner

# Detect Floating Point Number
# Enter your code here. Read input from STDIN. Print output to STDOUT


    
import re
def is_float(number):
    pattern = r"^[\+\-]?\d*\.\d+$"
    if re.match(pattern, number):
        try:
            float(number)
            return True
        except ValueError:
            return False
    return False
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        number = input().strip()
        print(is_float(number))

# Re.split()
regex_pattern = r"[,.]"

# Group(), Groups() & Groupdict()
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
m = re.search(r"([a-zA-Z0-9])\1+", input())
if m:
    print(m.group(1))
else:
    print(-1) 

# Re.findall() & Re.finditer()
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
pattern = r'(?<=[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ])[aeiouAEIOU]{2,}(?=[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ])'
m = re.findall(pattern, input())
if m:
    print("\n".join(m))
else:
    print(-1)

# Re.start() & Re.end()
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
S = input() 
pattern = input()  
found = False
start = 0
while True:
    m = re.search(pattern, S[start:])
    if not m:
        break
    found = True
    print(f"({m.start() + start}, {m.end() - 1 + start})")
    start += m.start() + 1
if not found:
    print("(-1, -1)")

# Regex Substitution
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
N = int(input())
pattern1 = r'(?<=\s)\|\|(?=\s)' 
pattern2 = r'(?<=\s)&&(?=\s)' 
for _ in range(N):
    text = input()
    text = re.sub(pattern2, "and", text)
    text = re.sub(pattern1, "or", text)
    print(text)

# Validating Roman Numerals

regex_pattern = r'M{0,3}(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[VX]|V?I{0,3})$'


# Validating phone numbers
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
N= int(input())
pattern = r'^(7|8|9)\d{9}$'
for i in range (N):
    match = str(bool(re.match(pattern, input())))
    if match == "True" :
        print ("YES")
    else:
        print ("NO" )

# Validating and Parsing Email Addresses
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
import email.utils
# Pattern per l'email valido
pattern = r'^[a-zA-Z][a-zA-Z0-9_.+-]*@[a-zA-Z]+\.[a-zA-Z]{1,3}$'
n = int(input())
for _ in range(n):
    entry = input().strip()
    name, mail = email.utils.parseaddr(entry)
    
    # Verifica se l'indirizzo email corrisponde al pattern
    if re.match(pattern, mail):
        print(email.utils.formataddr((name, mail)))

    


# Hex Color Code
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
pattern = r'#[a-fA-F0-9]{3,6}'
n = int(input())
for _ in range(n):
    line = input()
    if line.startswith("#"):
        continue
    matches = re.findall(pattern, line)
    for match in matches:
        print(match)

# Arrays

def arrays(arr):
     list1 =numpy.array(arr, float)
     return(list1[::-1])
     


# Shape and Reshape
import numpy

list1 = list(map(int, input().split()))
arr = numpy.array(list1)
print (numpy.reshape(arr,(3,3)))

# Transpose and Flatten
import numpy as np

N, M=map(int, input().split())
matrix =[]
for i in range (N):
    line = list(map(int, input().split()))
    matrix.append(line)

matrix = np.array(matrix)
print (np.transpose(matrix))
print (matrix.flatten())

# Concatenate
import numpy as np

N, M, P=map(int, input().split())
matrix1 =[]
for i in range(N):
     line = list(map(int, input().split()))
     matrix1.append(line)
matrix2 =[]
for i in range (M):
    lines = list(map(int, input().split()))
    matrix2.append(lines)

array_1 = np.array(matrix1)
array_2 = np.array(matrix2) 
print (np.concatenate((array_1, array_2), axis = 0) )

# Zeros and Ones
# Enter your code here. Read input from STDIN. Print output to STDOUT
import numpy as np

n =list(map(int, input().split()))
n = np.array(n)
print (np.zeros(n, dtype = np.int))
print (np.ones(n, dtype = np.int))

# Eye and Identity
# Enter your code here. Read input from STDIN. Print output to STDOUT
import numpy as np
np.set_printoptions(legacy='1.13')
N, M=map(int, input().split())
print (np.eye(N, M, k = 0))

# Array Mathematics
# Enter your code here. Read input from STDIN. Print output to STDOUT
import numpy as np

N, M=map(int, input().split())
matrix1=[]
for i in range (N):
    a = list(map(int, input().split()))
    matrix1.append(a)
a = np.array(matrix1)
matrix2=[]
for i in range (N):
    b = list(map(int, input().split()))
    matrix2.append(b)
b = np.array(matrix2)
    
    
print(np.add(a, b).reshape(N,M))
print(np.subtract(a,b).reshape(N,M))
print(np.multiply(a,b).reshape(N,M))
print((a // b).reshape(N,M))
print(np.mod(a,b).reshape(N,M))
print(np.power(a,b).reshape(N,M))



# Floor, Ceil and Rint
import numpy as np
np.set_printoptions(legacy='1.13')

A =list(map( float, input().split()))
a = np.array(A)
print (np.floor(a))
print (np.ceil(a))
print (np.rint(a))

# Sum and Prod
import numpy as np

np.set_printoptions(legacy='1.13')
N, M=map(int, input().split())
matrix1=[]
for i in range (N):
    a = list(map(int, input().split()))
    matrix1.append(a)
a = np.array(matrix1)

b = np.sum(a,axis=0)
b = np.array(b)
print(np.prod(b))


# Min and Max
import numpy as np

np.set_printoptions(legacy='1.13')

N, M=map(int, input().split())

matrix1=[]
for i in range (N):
    a = list(map(int, input().split()))
    matrix1.append(a)
a = np.array(matrix1)

b = np.min(a,axis=1)
b = np.array(b)
print(np.max(b))

# Mean, Var, and Std
import numpy as np
N, M=map(int, input().split())
matrix1=[]
for i in range (N):
    a = list(map(int, input().split()))
    matrix1.append(a)
a = np.array(matrix1)

print(np.mean(a, axis=1))
print(np.var(a, axis=0))
b= np.std(a)
print(round(b,11))

# Dot and Cross
import numpy as np

N = int(input())
matrix1=[]
for i in range (N):
    a = list(map(int, input().split()))
    matrix1.append(a)
a = np.array(matrix1)

matrix2=[]
for i in range (N):
    b = list(map(int, input().split()))
    matrix2.append(b)
b = np.array(matrix2)

c=np.dot(a,b)
print(c)

# Inner and Outer
import numpy as np
a = list(map(int, input().split()))
b = list(map(int, input().split()))
a=np.array(a)
b=np.array(b)
print(np.inner(a,b))
print(np.outer(a,b))

# Polynomials
import numpy as np

a = list(map(float,input().split()))
b=int(input())
print (np.polyval(a, b))

# Linear Algebra
import numpy as np

N = int(input())
matrix1=[]
for i in range (N):
    a = list(map(float, input().split()))
    matrix1.append(a)
    
    
b=  np.linalg.det(matrix1)   
print(round(b,2))


# HTML Parser - Part 1
# Enter your code here. Read input from STDIN. Print output to STDOUT
from html.parser import HTMLParser
N = int(input())
lines = [input() for _ in range(N)]
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Start :", tag)
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1] if attr[1] is not None else 'None'}")
    def handle_endtag(self, tag):
        print("End   :", tag)
    def handle_startendtag(self, tag, attrs):
        print("Empty :", tag)
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1] if attr[1] is not None else 'None'}")
parser = MyHTMLParser()
for line in lines:
    parser.feed(line)

# HTML Parser - Part 2
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if (data.find('\n') != -1):
            print(">>> Multi-line Comment")
        else:
            print(">>> Single-line Comment")
        print(data)
    def handle_data(self, data):
        if data.strip():  
            print(">>> Data")
            print(data)
  

  
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()

# Detect HTML Tags, Attributes and Attribute Values
# Enter your code here. Read input from STDIN. Print output to STDOUT
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1]}")
    def handle_startendtag(self, tag, attrs):
        print(tag)
        for attr in attrs:
            print(f"-> {attr[0]} > {attr[1]}")

html = ""      
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
parser = MyHTMLParser()
parser.feed(html)

# Validating UID
# Enter your code here. Read input from STDIN. Print output to STDOUT

#It must contain at least 2 uppercase English alphabet characters.
#It must contain at least 3 digits (0 - 9).
#It should only contain alphanumeric characters (a - z, A - Z & 0 - 9).
#No character should repeat.
#There must be exactly 10 characters in a valid UID.
import re
n = int(input())
for i in range(n):
    UID = input()
    a = len(re.findall(r"[A-Z]", UID)) >= 2
    b = len(re.findall(r"[0-9]", UID)) >= 3
    c = bool(re.match(r"^[a-zA-Z0-9]+$", UID))  
    d = len(set(UID)) == len(UID)  
    e = len(UID) == 10  
    if all([a, b, c, d, e]):
        print("Valid")
    else:
        print("Invalid")


# Validating Credit Card Numbers
# Enter your code here. Read input from STDIN. Print output to STDOUT

import re
pattern = r"^(4|5|6)\d{3}(-?\d{4}){3}$"
no_repeated_digits = r"(?!.*(\d)(-?\1){3})"
n = int(input())
for i in range(n):
    card = input()
    a = len(re.findall(r"[0-9]", card)) == 16
    b = bool(re.match(pattern, card))
    c = bool(re.match(no_repeated_digits, card))
    if all([a, b, c]):
        print("Valid")
    else:
        print("Invalid")

# Validating Postal Codes
regex_integer_in_range = r"^[1-9]\d{5}$"
regex_alternating_repetitive_digit_pair =r"(\d)(?=\d\1)"

# Matrix Script
#!/bin/python3
import math
import os
import random
import re
import sys


first_multiple_input = input().rstrip().split()
n = int(first_multiple_input[0])
m = int(first_multiple_input[1])
matrix = []
for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)
lista = []
for i in range(m):
    for e in range(n):
        lista.append(matrix[e][i])
string = ''.join(lista)
pattern = r"([a-zA-Z0-9])([^a-zA-Z0-9]+)([a-zA-Z0-9])"
def replace_special_characters(match):
    return f"{match.group(1)} {' '}{match.group(3)}"
result = re.sub(pattern, replace_special_characters, string)
print(result)

# Birthday Cake Candles
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#
def birthdayCakeCandles(candles):
    a = max(candles)
    b = candles.count(a)
    return b
    # Write your code here
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    candles_count = int(input().strip())
    candles = list(map(int, input().rstrip().split()))
    result = birthdayCakeCandles(candles)
    fptr.write(str(result) + '\n')
    fptr.close()

# Number Line Jumps
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#
def kangaroo(x1, v1, x2, v2):
    if v2>v1:
        while x2<= x1:
            if x1+v1 == x2+v2:
                return "YES"
            else:
                x1 = x1+v1
                x2 = x2+v2
        else:
            return "NO"
    elif v2<v1:
        while x1<= x2:
            if x1+v1 == x2+v2:
                return "YES"
            else:
                x1 = x1+v1
                x2 = x2+v2
        else:
            return "NO"
    else:
        if x1>x2 or x2>x1:
            return "NO"
        else:
            return "YES"
            

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    x1 = int(first_multiple_input[0])
    v1 = int(first_multiple_input[1])
    x2 = int(first_multiple_input[2])
    v2 = int(first_multiple_input[3])
    result = kangaroo(x1, v1, x2, v2)
    fptr.write(result + '\n')
    fptr.close()

# Viral Advertising
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#
def viralAdvertising(n):
    day = 1
    shared = 5
    liked = 2
    cumulative = 2
    for i in range(n-1):
        day += 1
        shared = liked * 3
        liked = shared//2
        cumulative = cumulative + liked
    return cumulative
        
        
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input().strip())
    result = viralAdvertising(n)
    fptr.write(str(result) + '\n')
    fptr.close()

# Recursive Digit Sum

#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'superDigit' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k
#
def superDigit(n, k):
    n = sum([int(n[digit]) for digit in range(len(n))]) * k
    return superDigit(str(n), 1) if n > 9 else n

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    n = first_multiple_input[0]
    k = int(first_multiple_input[1])
    result = superDigit(n, k)
    fptr.write(str(result) + '\n')
    fptr.close()

# Insertion Sort - Part 1
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'insertionSort1' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#
def insertionSort1(n, arr):
    r = arr[n-1]
    i=1
    while r<arr[n-1-i] and i<n:
            arr[n-i] = arr[n-i-1]
            print(*arr)
            i += 1
    arr[n-i] = r
    print(*arr)
        
            
if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    insertionSort1(n, arr)

# Insertion Sort - Part 2
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'insertionSort2' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort2(n, arr):
    for i in range(1,n):
        r = arr[i]
        j=i-1
        while j >=0 and arr[j]>r:
            arr[j+1] = arr[j]
            j-=1
        arr[j+1]=r
        print(*arr)

if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    insertionSort2(n, arr)

