# --->  1. count the character
# a = "helloworld"
# char_count = {}
#
# for char in a:
#     if char in char_count:
#         char_count[char] += 1
#     else:
#         char_count[char] = 1
# print(">>", char_count)
# result = ""
# for char, count in char_count.items():
#     result += char + str(count)
# print(result)


# -----> 2.
# s = 'spam, eggs, & ham'
# f = s.find('Eggs')
# print(">>", f)

# ----> 3. Print pattern
# *
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


# ------> 4. print number


# 1
# 12
# 123
# 1234
# 12345
# 1234
# 123
# 12
# 1
# for i in range(1, 6):
#     for j in range(1, i + 1):
#         print(j, end=" ")
#     print()
# for i in range(4, 0, -1):
#     for j in range(1, i + 1):
#         print(j, end=" ")
#     print()

# ------> 5. reverse num using for loop

# data = [1, 2, 3, 4, 5, 6, 7]
# reverse = []
# for i in range(len(data) - 1, -1, -1):
#     reverse.append(data[i])
# print(reverse)

# ------> 6. find index position and number of max number from list
# my_list = [1, 2, 3, 4, 190, 2, 0, 7, 22]
# max_num = my_list[0]
# num = ""
# for i in range(len(my_list)):
#     if my_list[i] > max_num:
#         max_num = my_list[i]
#         num = i
#
# print(f"max number is {max_num} of position {num}")

# ------> 7. binary number into decimal
# binary_number = "10011"
# decimal_num = 0
# for digit in binary_number:
#     decimal_num = decimal_num * 2 +int(digit)
# print(decimal_num)

#  binary to number
# binary_number = 102131
# decimal_number = int(binary_number, 2)
# # print(decimal_number)


#  ---------> 8.reverse the number without inbuilt function and reverse slicing
# # start = 0
# # end = len(number)-1
# # # print(end)
# # while start < end:
# #     number[start], number[end] = number[end], number[start]
# #     start += 1
# #     end -= 1
# # print(number)

#  ------> 9. Python Program to calculate the square root of a given number.
# # number = 144
# # # if number > 0:
# # guess_number = number / 2
# # while True:
# #     new_guess = (guess_number + number / guess_number) / 2
# #     if int(new_guess) == int(guess_number):
# #         guess = new_guess
# #         break
# #     guess_number = new_guess
# # print(guess)

#  ---------> 10. Python Program to calculate the power without using POW function.(using while loop).
# # num = 3
# # base = 4
# # result = 1
# # while base > 0:
# #     result *= num
# #     base -= 1
# # print(result)

# ----------> 11. Python Program to find Smallest number among three.
# # arr = [1, 3, 2, 4, 2, 4, 2, 3, 4, 7, 8, 6, 11, 0]
# # smallest_num = arr[0]
# # for num in arr:
# #     if num < smallest_num:
# #         smallest_num = num
# # print(smallest_num)

# -------> 12. Python Program to print Prime Number and not pm  in a given range.
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

#  ----------> 13. Python program to print first n Prime Number with explanation.
# number = 10
# pm = []
# for i in range(2, number):
#     if number % i == 0:
#         pm.append(i)
# print(pm)

# --------> 14. Write a program in Python to find given number is perfect or not?
# # -> perfect number is those number which having some positive integer which is equal to its positive division
#
# # a = 6
# # # div = []
# # added_num = 0
# # for num in range(1, a):
# #     if a % num == 0:
# #         # div.append(num)
# #         # print(div)
# #         added_num += num
# # if added_num == a:
# #     print("perfect number")
# # else:
# #     print("not perfect number")

# ---------> count the char that how many times they repeat
# def lengthOfLongestSubstring(s: str) -> str:
#     char_map = {}
#     for char in s:
#         if char in char_map:
#             char_map[char] += 1
#         else:
#             char_map[char] = 1
#     return char_map
#
#
# print(lengthOfLongestSubstring("abcabcbb"))


# --------------> count the char and return like this a4s5r2f3d1

# def countchar(char: str):
#     data = {}
#     count = ""
#     for i in char:
#         if i in data:
#             data[i] +=1
#         else:
#             data[i] = 1
#     for i, j in data.items():
#         count +=f"{i}{j}"
#     return count
#
# print(countchar("asaandanmeefdsd"))

