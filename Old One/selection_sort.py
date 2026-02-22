# number = "213124141"
# number_list = list(number)
#
# new_number = []
#
# for i  in range(len(number_list)-1):
#

# b = 30
#
#
# def fun(a, b=b):
#     return a + b
#
#
# print(fun(12))

def swap(a, b):
    a, b = b, a
    return a, b


a = 10
b = 20
a, b = swap(a, b)
print(f"after swap  a is {a} and b is {b}")
