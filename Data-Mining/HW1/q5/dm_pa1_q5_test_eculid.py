from operator import itemgetter
from statistics import mean

def generate_user_data(file_name):
    print("genearate_user_data")
    dist_file = open(file_name, "r")
    lines = dist_file.readlines()
    line_split = [list(w.split()) for w in lines[:-1]]
    user_movie_review = generate_dict(line_split, 0, 1)
    movie_user_review = generate_dict(line_split, 1, 0)
    return user_movie_review, movie_user_review
    #print (len(user_movie_review.keys()))
    #print (len(movie_user_review.keys()))

def generate_user_profile():
    dist_file = open("u.user","r")
    lines =dist_file.readlines()
    user_profile = {}
    for w in lines[:-1]:
        x = w.split("|")
        user_profile[x[0]] = [x[1],x[2],x[3]]
    return user_profile

def generate_dict(line_split, placeholder1, placeholder2):
    print("generate_dict")
    sample = {}
    for list in line_split:
        if list[placeholder1] not in sample.keys():
            sample[list[placeholder1]] = {list[placeholder2]:list[2]}
        else:
            sample[list[placeholder1]][list[placeholder2]] = list[2]
    return sample

def mean_find(user_movie_review, movie_user_review, movie):
    total = 0
    i =0
    mean = 0
    for user_reviews in movie_user_review.keys():
        if user_reviews == movie:
            for rater in user_movie_review.keys():
                if rater in movie_user_review[movie].keys():
                    i += 1
                    total += int(movie_user_review[movie][rater])
        if i != 0:
            mean = round(total/i)
    return mean

def avg(user_movie_review, movie_user_review, test_user_movie_review):
    predicted_user_movie_reviews = {}
    difference = []
    for user in test_user_movie_review.keys():
        predicted_user_movie_reviews[user] = {}
        for movie in test_user_movie_review[user].keys():
            #print(predicted_user_movie_reviews[user])
            if (movie not in user_movie_review[user].keys()):
                if(user not in predicted_user_movie_reviews.keys()):
                    mean_all = mean_find(user_movie_review, movie_user_review, movie)
                    predicted_user_movie_reviews[user] = {movie:mean_all}
                #round(mean(int(rating) for user_reviews in movie_user_review[movie] for rating in user_reviews))}
                else:
                    mean_all = mean_find(user_movie_review, movie_user_review,movie)
                    predicted_user_movie_reviews[user].update({movie :mean_all}) #round(mean(int(rating) for user_reviews in movie_user_review[movie] for rating in user_reviews))})
            difference.append(abs(int(test_user_movie_review[user][movie]) - int(predicted_user_movie_reviews[user][movie])))
    result = mean(difference)
    return result

def curr_movie_watched(user_movie_review, movie_user_review, curr_movie):
    curr_movie_watched_list = []
    for movie in movie_user_review.keys():
        if movie == curr_movie:
            for user in movie_user_review[movie].keys():
                curr_movie_watched_list.append(user)
            return curr_movie_watched_list

def rating_diff_gen(intersection_watched_list, curr_user, user, movie_user_review):
    distance = 0
    for movie in intersection_watched_list:
        distance += abs(int(movie_user_review[movie][user]) - int(movie_user_review[movie][curr_user]))**2
    return distance**0.5

def estimate(test_user, test_movie, user_movie_review,movie_user_review):
    rating_difference_map = []
    test_user_watch_list = user_movie_review[test_user].keys()
    for curr_user in movie_user_review[test_movie].keys():
        #print(user)
        #curr_movie_watched_user_list = [watched_user for movie in movie_user_review.keys() if movie == curr_movie for watched_user in movie_user_review[movie].keys()]
        curr_user_watch_list = user_movie_review[curr_user].keys()
        intersection_watch_list = list(set(test_user_watch_list).intersection(set(curr_user_watch_list)))
        if len(intersection_watch_list) > 0:
            rating_difference = rating_diff_gen(intersection_watch_list, test_user, curr_user, movie_user_review)
            rating_difference_map.append([curr_user, rating_difference])

    if rating_difference_map != []:
        sorted_rating_diff_map = sorted(rating_difference_map, key=itemgetter(1))[:11]
        #print(sorted_euclid_map)
        if len(sorted(rating_difference_map, key=itemgetter(1))[:11]) > 10:
            return top_pick_rating(sorted_rating_diff_map, test_movie, movie_user_review)
    return mean_find(user_movie_review, movie_user_review, test_movie)

def top_pick_rating(sorted_rating_map, test_movie, movie_user_review):
    top_user_rating_sum = 0
    i = 0
    #print(len(sorted_euclid_map))
    for pick in sorted_rating_map:
        i += 1
        top_user_rating_sum += int(movie_user_review[test_movie][pick[0]])
    if i == 10:
        return round(top_user_rating_sum/i)
    else:
        return mean_find(user_movie_review, movie_user_review, test_movie)

def ed(user_movie_review, movie_user_review, test_user_movie_review, test_movie_user_review):
    predicted_user_movie_reviews = {}
    for user in test_user_movie_review.keys():
        predicted_user_movie_reviews[user] ={}
        #print(test_movie_user_review[user].keys())
        difference = []
        for movie in test_user_movie_review[user].keys():
            #print(predicted_user_movie_reviews[user])
            #matched_users_list = []
            #print(curr_user_watch_list)
            #predicted_user_movie_reviews[user]={movie : estimate(user, movie, user_movie_review, movie_user_review)}
            if movie in movie_user_review.keys():
                predicted_user_movie_reviews[user].update({movie : estimate(user, movie, user_movie_review, movie_user_review)})
                difference.append(abs(int(test_user_movie_review[user][movie]) - int(predicted_user_movie_reviews[user][movie])))
        result = mean(difference)
    return result

def predictor(user_movie_review, movie_user_review, test_user_movie_review, test_movie_user_review, predictor_type):
    result = 0
    if predictor_type == "AVG":
        result = avg(user_movie_review, movie_user_review, test_user_movie_review)
        return result
    elif predictor_type == "ED":
        result = ed(user_movie_review, movie_user_review, test_user_movie_review,test_movie_user_review)
        return  result

##user_movie_review, movie_user_review = generate_user_data("u1.base")#input("File name:"))
##test_user_movie_review,test_movie_user_review = generate_user_data("u1.test")

user_profile = generate_user_profile()

result = predictor(user_movie_review, movie_user_review, test_user_movie_review, test_movie_user_review, "AVG")#input("Predictor Type: Values(AVG)"))
print(result)
##result = predictor(user_movie_review, movie_user_review, test_user_movie_review, test_movie_user_review, "ED")#input("Predictor Type: Values(AVG)"))
##print(result)
#print(predicted_user_movie_reviews1)

#predicted_user_movie_reviews1 = predictor(user_movie_review, movie_user_review, test_user_movie_review, test_movie_user_review)
#file = open("u1_base_avg_review.DATA File", "w")
#for user in predicted_user_movie_reviews1.keys():
#    for movie in predicted_user_movie_reviews1[user].keys():
#        file.write(user+"   "+movie+"  "+str(predicted_user_movie_reviews1[user][movie])+"\n")
l = []
l[0] = ["u1.base", "u1.test"]
l[1] = ["u2.base", "u2.test"]
l[2] = ["u3.base", "u3.test"]
l[3] = ["u4.base", "u4.test"]
l[4] = ["u5.base", "u5.test"]

for i in l:
    user_movie_review, movie_user_review = generate_user_data(i[0])#input("File name:"))
    test_user_movie_review,test_movie_user_review = generate_user_data(i[1])
    result = predictor(user_movie_review, movie_user_review, test_user_movie_review, test_movie_user_review, "ED")#input("Predictor Type: Values(AVG)"))
    print(result)
