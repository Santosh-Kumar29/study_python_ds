# string
# 1. Python program to remove given character from String.
# str = "absdfksfkjs"
# repl = "a"
# print(str.replace(repl, " "))


# 2. Python Program to count occurrence of a given characters in string.(count the number provided)
# str = "aadamdbasjha"
# occ = "d"
# count = 0
# for i in range(len(str)):
#     if str[i] == occ:
#         count = count + 1
# print("total occ", count)

# 3. Python Program to check if two Strings are Anagram.(anagram is the char, which is in match both the char)
# char1 = "abcd"
# char2 = "dcba"
# if sorted(char1) == sorted(char2):
#     print("anagram")
# else:
#     print("not anagram")


# 4. String Palindrome program in python
# char1 = "abc12"
# char2 = char1[::-1]
# print(char2)
# if char1 == char2:
#     print("palind")
# else:
#     print("not")

# 5. Python program to check given character is vowel or consonant
# char = "e"
# vowel = "aeiou"
# if char in vowel:
#     print("vowel")
# else:
#     print("cons")

# 6. Python program to check given character is digit or not
# ch = list("12e3")
# if ch.isdigit():
#     print("ok")
# else:
#     print("no")
# if ch >= '0' and ch <= '9':
#     print("yes")
# else:
#     print("not")
# for char in ch:
#     if char <= "0" and char >= "9":
#         # chained_comparison - '0' <=char <= '9'
#         print("yes")
#         break
# else:
#     print("no")

# 7. Write a program in Python to check whether a number is palindrome or not using recursive method.

# def check_pal(number):
# check_number = int(str(number)[::-1])
# '''-> it is using list slicing'''
# if number == check_number:
#     return "Yes"
# else:
#     return "NO"
# reversed_num =
# while number > 0:
#     remainder = number % 10
#     reversed_num = ()


# print(check_pal(101))


# number = [12, 15, 13, 2, 4, 3]
# divide = [2, 5]
# addition_number = []
# min_num = []
# added_number = 0
# minus_number = 0
# count = 0
# for i in range(len(number)):
#     for j in range(len(divide)):
#         if number[i] % divide[j] == 0:
#             print(f"{number[i]} is divisible of {divide[j]}")
# print(added_number)


number = [2, 5, 4, 3, 6, 3]
added_number = [5, 5]
addedn = []
minusn = []
expon = []
for i in range(len(number)):
    added = 0
    minus = 0
    exp = number[i]
    for j in range(len(added_number)):
        added += added_number[j]
        minus -= added_number[j]
        exp **= added_number[j]
    addedn.append(number[i] + added)
    minusn.append(number[i] + minus)
    expon.append(exp)
print(addedn)
print(minusn)
print("expo", expon)
