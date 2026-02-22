# Encapsulation
# -> All the necessary data are bind together and all unnecessary details are hidden to the normal user

class BankAccount:
    def __init__(self, balance):
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        print(self.balance)

    def withdraw(self, amount):
        if amount > self.balance:
            print("Insufficient funds")
        else:
            self.balance -= amount

    def get_balance(self):
        return self.balance


acc = BankAccount(10)
acc.deposit(10)
acc.withdraw(50)