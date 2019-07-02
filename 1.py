def TuptoDict(tupe, d):
  for a, b in tupe:
    d.setdefault(a, []).append(b)
  return d

tup_input = [( "John", ("Physics", 80)), ("Daniel", ("Science", 90)),
("John", ("Science", 95)), ("Mark",("Maths", 100)), ("Daniel", ("History", 75)), ("Mark", ("Social", 95))]

dict1 = {}
dict2 = {}

dict2 = TuptoDict(tup_input, dict1)
print("output before sorting : ", dict2)

dict_values = list(dict2.values())

list1 = []

# soring values based on subject names
for i in dict_values:
    sorted_list = sorted(i, key=lambda item: item[0],reverse=False)
    list1.append(sorted_list)

k=0
# reassigning sorted values to the keys of dictionary
for key in dict2:
    dict2[key] = list1[k]
    k=k+1
print("final output after sorting : ", dict2)


