# Write a program that add all the marks of students for given subject. Each element of the list is array of marks
# L1 = [[90,60,30],[69,80,79],[89,90,60]]

L1 = [[90,60,30],[69,80,79],[89,90,60]]

# creating a empty list of lists for all subjects: [ [subject1], [subject2], [subject3] ]
subjects = [[] for i in range(3)]

# outer for loop: looping for every student
for student_no in range(len(L1)):
    # inner for loop: looping with the student for marks
    for subject_no in range(len(L1[student_no])):
        subjects[subject_no].append(L1[student_no][subject_no])
        # print(subjects[subject_no])

for subject in subjects:
    print(sum(subject))