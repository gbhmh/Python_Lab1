class Flight:
    def __init__(self, f_name, f_num):
        self.f_name = f_name
        self.f_num = f_num

    def flight_display(self):
        print('flight_name: ', self.f_name)
        print('flight_number: ', self.f_num)

class Employee:
    def __init__(self, e_id, e_name, e_age, e_gender):
        self.e_name = e_name
        self.e_age = e_age
        self.__e_id = e_id #private variable
        self.e_gender =  e_gender
    def emp_display(self):
        print("employee_name: ",self.e_name)
        print('employee_id: ', self.__e_id)
        print('employee_age: ',self.e_age)
        print('employee_gender: ', self.e_gender)

class Passenger:
    def __init__(self):
        Passenger.__passport_number = input("enter the passport number of passenger: ") #private variable
        Passenger.name = input('enter name of passenger: ')
        Passenger.age = int(input('enter age of passenger: '))
        Passenger.gender = input('enter the gender: ')
        Passenger.class_type = input('select b for business or e for economy class: ')

class Luggage():
    bag_fare = 0
    def __init__(self, checkin_bags):
        self.checkin_bags = checkin_bags
        if checkin_bags > 2 :
            self.bag_fare = (checkin_bags-2)*40
        else:
            pass

class Fare(Luggage): #inheritance
    offline = 250
    online = 300
    total_fare=0
    def __init__(self):
        super().__init__(int(input('enter number of check-in bags carrying other than cabin bag : '))) #super call
        x = input('buy ticket through online or offline: ')
        if x == 'online':
            Fare.total_fare = self.online + self.bag_fare
        elif x == 'offline':
            Fare.total_fare = self.offline + self.bag_fare
        else:
            pass

class Ticket(Passenger, Fare): #inheritance
    def __init__(self):
        print("Passenger name:",Passenger.name)
        if Passenger.class_type == "b":
            str = "business"
            Fare.total_fare+=80
        else:
            str = "economy"
            pass

        print("Passenger class type:",str)
        print("Total fare:",Fare.total_fare)


f1=Flight('AA','AA786J')
f1.flight_display()

emp1 = Employee('e1','Gopi Chand',24,'M')
emp1.emp_display()

p1 = Passenger()

fare1=Fare()

t= Ticket()