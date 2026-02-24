# Python Interview Questions — Answers

---

## 1. Maximum Sum of Subarray of Size K (Sliding Window)

```python
arr = [1, 2, 3, 4, 5, 6]
k = 3

def max_sum_subarray(arr, k):
    if len(arr) < k:
        return -1

    # Compute sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum

    # Slide the window: subtract the element going out, add the element coming in
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)

    return max_sum

print(max_sum_subarray(arr, k))  # Output: 15 (4+5+6)
```

**Key Concept:** Sliding Window technique — O(n) time, O(1) space. Instead of recalculating the sum for every window, we slide by removing the leftmost element and adding the new rightmost element.

---

## 2. Comparing Two Strings for Anagrams

```python
str1 = "Listen"
str2 = "Silent"

def are_anagrams(s1, s2):
    return sorted(s1.lower()) == sorted(s2.lower())

print(are_anagrams(str1, str2))  # Output: True
```

**Alternative (using dict frequency count — more efficient):**

```python
def are_anagrams(s1, s2):
    def char_count(s):
        count = {}
        for c in s.lower():
            count[c] = count.get(c, 0) + 1
        return count
    return char_count(s1) == char_count(s2)

print(are_anagrams(str1, str2))  # Output: True
```

**Key Concept:** Two strings are anagrams if they contain the same characters with the same frequencies. `sorted()` is O(n log n), dict counting is O(n).

---

## 3. Find the Non-Repeating Character in a String

```python
str1 = "Listen"

def first_non_repeating(s):
    count = {}
    for char in s.lower():
        count[char] = count.get(char, 0) + 1
    for char in s.lower():
        if count[char] == 1:
            return char
    return None

print(first_non_repeating(str1))  # Output: 'l'
```

**Key Concept:** Use a dict to store frequency of each character, then iterate again to find the first character with count == 1. Time: O(n), Space: O(n).

---

## 4. Find the Second Largest Number in a List

```python
def second_largest(nums):
    first = second = float('-inf')
    for n in nums:
        if n > first:
            second = first
            first = n
        elif n > second and n != first:
            second = n
    return second if second != float('-inf') else None

print(second_largest([10, 20, 4, 45, 99]))  # Output: 45
```

**Key Concept:** Single pass O(n) solution. Track both the largest and second largest. Avoid using `sort()` which would be O(n log n).

---

## 5. Find Common Elements Between Two Lists

```python
def common_elements(list1, list2):
    return list(set(list1) & set(list2))

print(common_elements([1, 2, 3, 4], [3, 4, 5, 6]))  # Output: [3, 4]
```

**Alternative (using list comprehension):**

```python
def common_elements(list1, list2):
    set2 = set(list2)
    return [x for x in list1 if x in set2]

print(common_elements([1, 2, 3, 4], [3, 4, 5, 6]))  # Output: [3, 4]
```

**Key Concept:** Using sets gives O(n + m) time complexity. The intersection operator `&` finds common elements efficiently.

---

## 6. Find the Longest Word in a Sentence

```python
def longest_word(sentence):
    words = sentence.split()
    return max(words, key=len)

print(longest_word("The quick brown fox jumped"))  # Output: 'jumped'
```

**Key Concept:** `split()` breaks the sentence into words, `max()` with `key=len` finds the longest one. Time: O(n).

---

## 7. Decorator in Python

```python
# A decorator is a function that takes another function as an argument,
# adds some functionality, and returns a modified function — without
# changing the original function's source code.

def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before the function call")
        result = func(*args, **kwargs)
        print("After the function call")
        return result
    wrapper.__name__ = func.__name__  # Preserve original function name
    wrapper.__doc__ = func.__doc__    # Preserve original docstring
    return wrapper

@my_decorator
def say_hello(name):
    """Greets a person by name."""
    print(f"Hello, {name}!")

say_hello("Santosh")
# Output:
# Before the function call
# Hello, Santosh!
# After the function call
```

**Real-World Example — Logging Decorator:**

```python
def logger(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper

@logger
def add(a, b):
    return a + b

add(3, 5)
# Output:
# Calling add with args=(3, 5), kwargs={}
# add returned 8
```

**Key Points:**
- `@decorator` is syntactic sugar for `func = decorator(func)`
- Manually copy `__name__` and `__doc__` to preserve the original function's metadata
- Common use cases: logging, authentication, caching, rate limiting, retry logic

---

## 8. Two Sum — Return Indices That Add Up to Target

```python
def two_sum(nums, target):
    seen = {}  # value -> index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

print(two_sum([2, 7, 11, 15], 9))   # Output: [0, 1]
print(two_sum([3, 2, 4], 6))        # Output: [1, 2]
```

**Key Concept:** Hash map approach — O(n) time, O(n) space. For each number, check if its complement (target - num) has already been seen. This is the classic LeetCode #1 problem.

---

## 9. Longest Substring Without Repeating Characters

```python
def length_of_longest_substring(s):
    char_index = {}  # character -> last seen index
    left = 0
    max_length = 0

    for right, char in enumerate(s):
        if char in char_index and char_index[char] >= left:
            left = char_index[char] + 1
        char_index[char] = right
        max_length = max(max_length, right - left + 1)

    return max_length

print(length_of_longest_substring("abcabcbb"))  # Output: 3 ("abc")
print(length_of_longest_substring("bbbbb"))      # Output: 1 ("b")
print(length_of_longest_substring("pwwkew"))     # Output: 3 ("wke")
```

**Key Concept:** Sliding Window with a hash map. Maintain a window `[left, right]` with no duplicates. When a duplicate is found, move `left` past the previous occurrence. Time: O(n), Space: O(min(n, m)) where m is charset size. LeetCode #3.

---

## 10. Group Anagrams

```python
def group_anagrams(strs):
    anagram_map = {}
    for word in strs:
        key = ''.join(sorted(word))  # sorted characters as key
        if key not in anagram_map:
            anagram_map[key] = []
        anagram_map[key].append(word)
    return list(anagram_map.values())

print(group_anagrams(["eat","tea","tan","ate","nat","bat"]))
# Output: [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
```

**Alternative (using tuple of character counts as key):**

```python
def group_anagrams(strs):
    anagram_map = {}
    for word in strs:
        count = [0] * 26
        for char in word:
            count[ord(char) - ord('a')] += 1
        key = tuple(count)
        if key not in anagram_map:
            anagram_map[key] = []
        anagram_map[key].append(word)
    return list(anagram_map.values())
```

**Key Concept:** All anagrams produce the same sorted string (or same character frequency count). Use that as a hash map key. Time: O(n * k log k) for sorted approach, O(n * k) for count approach, where n = number of strings, k = max string length. LeetCode #49.

---

## 11. Valid Palindrome (Ignoring Non-Alphanumeric Characters)

```python
def is_palindrome(s):
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]

print(is_palindrome("A man, a plan, a canal: Panama"))  # Output: True
print(is_palindrome("race a car"))                       # Output: False
```

**Two-Pointer Approach (O(1) extra space):**

```python
def is_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if s[left].lower() != s[right].lower():
            return False
        left += 1
        right -= 1
    return True

print(is_palindrome("A man, a plan, a canal: Panama"))  # Output: True
```

**Key Concept:** Strip non-alphanumeric characters, compare from both ends. The two-pointer approach avoids creating a new string, giving O(1) space. LeetCode #125.
