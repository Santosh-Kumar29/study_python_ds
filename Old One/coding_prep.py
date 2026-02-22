# string
# 1. Python program to remove given character from String.
# str = "absdfksfkjs"
# repl = "a"
# print(str.replace(repl, " "))


# 2. Python Program to count occurrence of a given characters in string.(count the number provided)
# str = "aadamdbasjha"
# occ = "a"
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

# print((1, 2, 3,1))

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
#
#
# print(check_pal(101))


# 8. Write a program in Python to check if a number is binary?

# number = 101101
# flag = False
# for digit in str(number):
#     if digit != '0' and digit != '1':
#         flag = False
#     else:
#         flag = True
# if flag:
#     print("Binary number")
# else:
#     print("Not Binary Number")


# convert number into binary number
# number = 25
# binary = ""
# while number > 0:
#     binary = str(number % 2) + binary
#     number = number // 2
# print(binary)

# convert binary into number
# binary = "11001"
# number = int(binary, 2)
# print(number)

# 10. Write a program in Python to find sum of digits of a number using recursion?

# a = 10
# b = 11
# a = a + b
# print(a)

# 11. Write a program in Python to swap two numbers without using third variable?
# a = 11
# b = 10
# a, b= b, a
# print(a)
# print(b)

# 12. Write a program in Python to swap two numbers using third variable

# a = 12
# b = 34
# c = a
# a = b
# b = c
# print(a)
# print(b)

# # 13. Write a program in Python to find prime factors of a given integer.
# number = 12
# factor = []
# div = 2
# while div <= number:
#     if number % div == 0:
#         factor.append(div)
#         number = number / div
#     else:
#         div += 1
# print(factor)

# 14. Write a program in Python to add two integer without using arithmetic operator?
# a = 12
# b = 23
# while b != 0:
#     c = a & b
#     a = a ^ b
#     b = c << 1
# print(a)


#  15. Write a program in Python to find given number is perfect or not?
# -> perfect number is those number which having some positive integer which is equal to its positive division

# a = 6
# # div = []
# added_num = 0
# for num in range(1, a):
#     if a % num == 0:
#         # div.append(num)
#         # print(div)
#         added_num += num
# if added_num == a:
#     print("perfect number")
# else:
#     print("not perfect number")


# 16. Python Program to find the Average of numbers with explanations.
# numbers = [5, 8, 2, 10, 6]
# sum_num = 0
# for num in numbers:
#     sum_num += num
# print("sum", sum_num)
# print("length", len(numbers))
# print("average", int(sum_num) / len(numbers))

# 17. Python Program to calculate factorial using iterative method.
# number = 5
# for num in range(1, number):
#     number *= num
# print(number)

# 19. Python Program to check a given number is even or odd.
# number = int(input())
# if number % 2 == 0:
#     print("even number")
# else:
#     print("odd num")

# 20. Python program to print first n Prime Number with explanation.
# number = 10
# print_num = []
# for i in range(2, number):
#     if number % i == 0:
#         print_num.append(i)
# print(print_num)

# 21. Python Program to print Prime Number in a given range.
# arr = [1, 3, 2, 4, 2, 4, 2, 3, 4, 7, 8, 6, 11]
# prime_num = []
# not_prime = []
#
# for num in arr:
#     if num < 2:
#         not_prime.append(num)
#     else:
#         flag = True
#         for i in range(2, int(num)):
#             if num % i == 0:
#                 not_prime.append(num)
#                 flag = False
#         if flag:
#             prime_num.append(num)
#
# print(prime_num)
# print(not_prime)

# 22. Python Program to find Smallest number among three.
# arr = [1, 3, 2, 4, 2, 4, 2, 3, 4, 7, 8, 6, 11, 0]
# smallest_num = arr[0]
# for num in arr:
#     if num < smallest_num:
#         smallest_num = num
# print(smallest_num)

# 23. Python program to calculate the power using the POW method.
# num = 3
# base = 4
# result = 1
# number = pow(num, base)
# print(">>", number)


# 24. Python Program to calculate the power without using POW function.(using for loop).
# for i in range(base):
#     result *= num
# print(result)

# 25. Python Program to calculate the power without using POW function.(using while loop).
# num = 3
# base = 4
# result = 1
# while base > 0:
#     result *= num
#     base -= 1
# print(result)

#  26. Python Program to calculate the square of a given number.
# number = 10
# number = number * number
# print(number)

# 27. Python Program to calculate the cube of a given number.
# number = 10
# print(number * (number * number))

# 28. Python Program to calculate the square root of a given number.
# number = 144
# # if number > 0:
# guess_number = number / 2
# while True:
#     new_guess = (guess_number + number / guess_number) / 2
#     if int(new_guess) == int(guess_number):
#         guess = new_guess
#         break
#     guess_number = new_guess
# print(guess)

# 29. Python program to calculate LCM of given two numbers.
# numer1 = 6
# number2 = 15
# max_num = max(numer1, number2)
# while True:
#     if max_num % numer1 == 0 and max_num % number2 == 0:
#         lcm = max_num
#         break
#     max_num += 1
# print(lcm)

# 30. Python Program to find GCD or HCF of two numbers.
# number1 = 60
# number2 = 90
# num = min(number1, number2)
# ns = []
# for i in range(2, num):
#     if number1 % i == 0 and number2 % i == 0:
#         ns.append(i)
#     i += 1
# new_num = ns[1]
# for new in ns:
#     if new > new_num:
#         new_num = new
# print(new_num)

# 31. Python Program to find GCD of two numbers using recursion.
# def get_gcd(num1, num2):
#     num = min(num1, num2)
#     gcd = 1
#     for i in range(2, num):
#         if num1 % i == 0 and num2 % i == 0:
#             # print(">>", gcd)
#             gcd = i
#     return gcd
#
#
# print(get_gcd(60, 90))

# 32. Python Program to Convert Decimal Number into Binary.

# number = 0.75
# binary = ""
# #
# # if number == 0:
# #     binary = "0"
#
# while number != 0:
#     number *= 2
#     if number >= 1:
#         binary += "1"
#         number -= 1
#     else:
#         binary += "0"
# print(binary)

# 33. Python Program to convert Decimal number to Octal number.

# number = 52
#
# octal = ""
#
# while number > 0:
#     print(number)
#     octal = str(number % 8) + octal
#     number = number // 8
# print(octal)

# 34. Python Program to check the given year is a leap year or not.
# year = 2001
# if (year % 400 == 0) and (year % 100 == 0):
#     print(f"{year} is a leap year")
# elif (year % 4 == 0) and (year % 100 != 0):
#     print(f"{year} is a leap year")
# else:
#     print(f"{year} is not a leap year")

# 35. Python Program to convert Celsius to Fahrenheit.
# celsius = 32
# farehinite = (celsius * 9 / 5) + 32
# print(farehinite)

# 36. Python Program to convert Fahrenheit to Celsius.
# farehinite = 77.7
# celsius = (farehinite - 32) * 5 / 9
# print(celsius)


#  37. Python program to calculate Simple Interest with explanation.

# principle = 100
# rate  =32
# time = 10
# si = (principle * rate * time) / 100
# print(si)

# =========================================================================================================================
# Some Extra Practice

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


# ****************************************************


# reverse the number without inbuilt function and reverse slicing
# start = 0
# end = len(number)-1
# # print(end)
# while start < end:
#     number[start], number[end] = number[end], number[start]
#     start += 1
#     end -= 1
# print(number)

# using list slicing
# new_number = number[::-1]
# print(new_number)


# any other method


# #
# number = [2, 5, 4, 3, 6, 3]
# added_number = [5, 5]
# addedn = []
# minusn = []
# expon = []
# for i in range(len(number)):
#     added = 0
#     minus = 0
#     exp = number[i]
#     for j in range(len(added_number)):
#         added += added_number[j]
#         minus -= added_number[j]
#         exp **= added_number[j]
#     addedn.append(number[i] + added)
#     minusn.append(number[i] + minus)
#     expon.append(exp)
# print(addedn)
# print(minusn)
# print("expo", expon)

# binary number into decimal
# binary_number = "10011"
# decimal_number = 0
# # for digit in binary_number:
# #     decimal_number = decimal_number * 2 + int(digit)
# # print(decimal_number)
#
# # decimal_number = int(binary_number, 2)
# # print(decimal_number)

######################################################################################################################

# 1. How to convert a list into a string?
# my_list = ['Hello', 'world', '!']
# new_str = ''
# for char in my_list:
#     new_str += char
# print(new_str)

# 2. How to convert a list into a tuple?

# my_list = [1, 2, 3, 4, 5]
# my_tuple = ()
# for num in my_list:
#     my_tuple += (num,)
# print(my_tuple)

# 3. How to convert a list into a set?
# my_list = [1, 2, 3, 4, 5]
# my_set = set()
#
# for num in my_list:
#     my_set.add(num)
#
# print(my_set)

# 4. How to count the occurrences of a particular element in the list?
# my_list = [1, 2, 3, 4, 5, 2]
# target = 2
# count = 0
# position = []
# for i in range(len(my_list)):
#     if target == my_list[i]:
#         count += 1
#         position.append(i)
# print(f"occ of {count} from index {position}")

# 5. How do you Concatenate Strings in Python?

# str1 = "Hello"
# str2 = "world!"
# new_concatenate = str1 + " " + str2
# print(new_concatenate)

# 6. find index position and number of max number from list
# my_list = [1, 2, 3, 4, 190, 2, 0, 7, 22]
# max_num = my_list[0]
# max_pos = ""
# for num in range(len(my_list)):
#     if my_list[num] > max_num:
#         max_num = my_list[num]
#         max_pos = num
# print(f"max number is {max_num} of position {max_pos}")
#
# for i, num in enumerate(my_list):
#     if num > max_num:
#         max_num = num
#         max_pos = i
#
#
# print(f"max number is {max_num} of position {max_pos}")


# data = lambda x: max(my_list)
# print(data(my_list))
# my_list = [1, 2, 3, 4, 5, 2, 0]
#
# sum_numbers = lambda lst: 0 if len(lst) == 0 else lst[0] + sum_numbers(lst[1:])
# result = sum_numbers(my_list)
# print(result)

# x = 10
# y = 15
# data = lambda x, y: x + y
# print(data(x, y))

# from functools import reduce
data = [1, 2, 3, 4, 5, 6, 7]

# # print(data[-2::-1])
# sum_num = reduce(lambda x, y: x + y, data)
# print(sum_num)
# data = [1, 2, 3, 4, 5, 6, 7]

# print(data[2:5])
# sum_num = lambda lst: sum(lst)
# result = sum_num(data)
# print(result)
# num = 0
# for i in data:
#     num +=i
# print(num)
# sum_num = lambda x: sum(x)
# result = sum_num(data)
# print(result)

# reverse num using for loop
# data = [1, 2, 3, 4, 5, 6, 7]
# reverse = []
# for i in range(len(data) - 1, -1, -1):
#     reverse.append(data[i])
# print(reverse)


# print (1
# 12
# 123
# 1234
# 12345
# 1234
# 123
# 12
# 1)
#
# for i in range(1, 6):
#     for j in range(1, i+1):
#         print(j, end='')
#     print()
# for i in range(4, 0, -1):
#     for j in range(1, i +1):
#         print(j, end='')
#     print()
# table = 12
# for i in range(1, 11):
#     print(f"{table} * {i} = {table * i}")

# I = [1, 2, 3, 4, 5]
# data = [x & 1 for x in I]
# print(data)

## Print *
# * *
# * * *
# * * * *
# * * * * *
# * * * * *
# * * * *
# * * *
# * *
# *

# for i in range(1, 6):
#     for j in range(1, i + 1):
#         print("*", end=" ")
#     print()
# for i in range(5, 0, -1):
#     for j in range(1, i + 1):
#         print("*", end=" ")
#     print()
#
# def add(*kwargs):
#     number = list(kwargs)
#     added_num = 0
#     for i in number:
#         added_num += i
#     print(added_num)
#
#
# print(add(1, 2, 4, 3, 2, 4))


# class First:
#
#     def func(self):
#         print("aaa")
#
#
# class Second(First):
#     def func(self):
#         print("bbb")
#         super().func()
#
#
# class Third(First):
#     def func(self):
#         super().func()
#
#
# class Four(Second, Third):
#     def fun2(self):
#         print("okk")
#
# object = Four()
# object.func()
#
#
# x = 1/0
# print(x)

# s = 'spam, eggs, & ham'
# f = s.find('Eggs')
# print(">???>", f)


