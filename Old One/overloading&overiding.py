# Method overloading
# method overloading is refers to a ability to define multiple method with the same name but different parameter
# class MyClass:
#     def my_method(self, a1, a2=None):
#         if a2 is None:
#             print("hello")
#             return "Some value"
#         else:
#             print("ar+a1")
#             return a1 + a2
#
#
# check = MyClass()
# print(check.my_method(2))


# Method Overiding
# method overiding is a ability to define a method in a subclass that already exist in a parent class
# class ParentClass:
#     def my_method(self):
#         print("Parent class method")
#
#
# class ChildClass(ParentClass):
#     def my_method(self):
#         print("Child class method")
#
#
# parent_obj = ParentClass()
# parent_obj.my_method()  # Output: Parent class method
#
# child_obj = ChildClass()
# child_obj.my_method()
