import math


# Get the input number from the user
# number = input("Enter a number: ")

# Calculate the square root
# square_root = math.sqrt(number)
#
# # Print the result
# print("The square root of", number, "is", square_root)
#
# # number = float(input("Enter a number: "))
# test = number / 2
# new_guess = (test + number / test) / 2
# print(">>>", float(new_guess))
# number = 10
# data = False
# for i in range(number + 1):
#     if i * i == number:
#         data = True
#         break
#     elif i * i > number:
#         print(i - 1)
#         break

# if not found_exact_match:
#     print(number)

# newton method
# number = 10
# divis = number/2
# new_number = divis-(divis * divis - number) / (2*divis)
# print(new_number)

def add(*args):
    num = list(args)
    data = 0
    for i in num:
        data += i
    return data


print(add(1, 2, 4, 3, 7))
