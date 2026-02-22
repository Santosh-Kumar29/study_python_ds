# num = int(input("Enter a number:"))
#
# flag = False
# for i in range(2, num):
#     if num % 2 == 0: break
#     flag = True
#
# if flag:
#     print(f"{num} is Prime number")
# else:
#     print(f"{num} is not prime number")


number_data = [1, 3, 2, 4, 2, 4, 2, 3, 4, 7, 8, 6, 11]
prime = []
# for num in number_data:
#     if num > 1:
#         prime_number = True
#         for i in range(2, int(num ** 0.5) + 1):
#             if num % i == 0:
#                 prime_number = False
#                 break
#         if prime_number:
#             prime.append(num)
#
# print(prime)


# for num in number_data:
#     if num > 1:
#         for i in range(2, num):
#             if num % i == 0:
#                 break
#         else:
#             prime.append(num)
# print(prime)


# n = 1
# flag = False
# for i in range(2, n):
#     if n % 2 == 0:
#         print("not prime number")
#         break
#     else:
#         print("prime")
#         break
# else:
#     print("prime number always greater then 1")

# arr = [1, 3, 2, 4, 7, 9]
# prime_num = []
# for data in arr:
#     is_prime = True
#     if data > 2:
#         for num in range(2, data):
#             if data % num == 0:
#                 is_prime = False
#     if is_prime:
#         prime_num.append(data)
# print(prime_num)
arr = [1, 3, 2, 4, 2, 4, 2, 3, 4, 7, 8, 6, 11]
not_prime = []
is_prime = []
even_num = []
odd_num = []
for num in arr:
    if num < 2:
        not_prime.append(num)
    elif num % 2 == 0:
        even_num.append(num)
    elif num % 2 != 0:
        odd_num.append(num)

    else:
        prime_data = True
        for i in range(2, num):
            if num % i == 0:
                not_prime.append(num)
                prime_data = False
                break
        if prime_data:
            is_prime.append(num)
print(not_prime)
print(is_prime)
print("even", even_num)
print("odd", odd_num)

# number = int(input("Enter a number"))
# flag = False
# for i in range(2, number):
#     if (number % i) == 0:
#         flag = True
#         break
#
# if flag:
#     print(f"{number} is not a prime number")
# else:
#     print(f"{number} is a prime number")
