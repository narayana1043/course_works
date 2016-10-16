import math
def merge_lists(list1, list2):
    merged_list = []
    list1.append(math.inf)
    list2.append(math.inf)
    a = list1.pop(0)
    b = list2.pop(0)
    while list1 or list2:
        if a <= b:
            merged_list.append(a)
            a = list1.pop(0)
        else:
            merged_list.append(b)
            b = list2.pop(0)
    return merged_list



def merge_sorted_lists(list_of_sorted_lists):
    length = len(list_of_sorted_lists)
    temp_list_of_lists = list()
    while length > 1:
        list1 = list_of_sorted_lists[-1]
        list2 = list_of_sorted_lists[-2]
        temp_list_of_lists.append(merge_lists(list1, list2))
        length = length - 2
        if length == 1:
            list1 = list_of_sorted_lists[0]
            temp_list_of_lists.append(list1)
    if len(temp_list_of_lists) == 1:
        return temp_list_of_lists
    return merge_sorted_lists(temp_list_of_lists)

print(merge_sorted_lists([[1,2,3],[4,5,6],[7,8,9],[10,11,12,13,14,15,17],[3,5,6,7,8]]))
