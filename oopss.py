# class ABC:
#     def __init__(self, name):
#         self.name = name
#
#     def get_name(self):
#         print("1111")
#         return self.name
#
#
# class AB(ABC):
#     def __init__(self, name):
#         super().__init__(name)
#
#     def provide_info(self):
#         print(f"name is {self.get_name()}")
#
#
# obj = AB("santosh")
# name = obj.get_name()
# print(name)


# class Bankacc:
#     def __int__(self, balance):
#         self.balance = balance
#
#     def deposit(self, amount):
#         self.balance += amount
#
#     def withdrawl(self, amount):
#         if amount > self.balance
#             print("insifficinet fund")
#         else:
#             self.balance -= amount
#
#     def get_balance(self):
#         return self.balance

# def add(a, b):
#     return a + b
#
#
# print(2, 4)
# print("aa", "bbb")

def decorator_function(original_function):
    def wrapper_function():
        print("Before the function is called.")
        original_function()
        print("After the function is called.")

    return wrapper_function


@decorator_function
def hello_world():
    print("Hello, world!")
