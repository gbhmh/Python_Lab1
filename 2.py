def long_substrs(str1):
  curr_list = []    #for holding current string
  sub_string_list = [] #for holding all sub strings

# for generating list of subtrings without repeating characters
  for letter in str1:
    if letter in curr_list:
      sub_string_list.append(''.join(curr_list))
      curr_pos = curr_list.index(letter) + 1
      curr_list = curr_list[curr_pos:]
      # print(curr_list)
    curr_list += letter
  sub_string_list.append(''.join(curr_list))
  # print(sub_string_list)

# for displaying only substrings that has maximum length of all substrings
  max_len = max(len(k) for k in sub_string_list)
  for k in sub_string_list:
    if len(k) == max_len:
      print((k, len(k)))

str = input("Enter a String : ")
long_substrs(str)

