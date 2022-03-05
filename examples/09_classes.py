class Employee:
    """Common base class for all employees"""

    empCount = 0

    def __init__(self, name, salary):
        print("creating")
        self.name = name
        self.salary = salary
        Employee.empCount += 1
        # type(self).empCount += 1

    def displayCount(self):
        print("Total Employee %d" % Employee.empCount)

    def displayEmployee(self):
        print("Name : ", self.name, ", Salary: ", self.salary)

    def __str__(self):
        return "Name : " + self.name + ", Salary: " + str(self.salary)

    def __del__(self):
        print("takin out the trash")
        type(self).empCount -= 1


emp2 = Employee("Jed", 5000)
emp1.name
