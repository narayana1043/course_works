from operator import itemgetter
from statistics import mean


def generate_user_data(file_name):
    # print("genearate_user_data")
    dist_file = open(file_name, "r")
    lines = dist_file.readlines()
    line_split = [list(w.split()) for w in lines[:-1]]
    user_movie_review = generate_dict(line_split, 0, 1)
    movie_user_review = generate_dict(line_split, 1, 0)
    return user_movie_review, movie_user_review
    # print (len(user_movie_review.keys()))
    # print (len(movie_user_review.keys()))


def generate_user_profile():
    dist_file = open("u.user", "r")
    lines = dist_file.readlines()
    user_profile = {}
    for w in lines[:-1]:
        x = w.split("|")
        user_profile[x[0]] = [x[1], x[2], x[3]]
    return user_profile


def generate_dict(line_split, placeholder1, placeholder2):
    # print("generate_dict")
    sample = {}
    for list in line_split:
        if list[placeholder1] not in sample.keys():
            sample[list[placeholder1]] = {list[placeholder2]: list[2]}
        else:
            sample[list[placeholder1]][list[placeholder2]] = list[2]
    return sample


def mean_find(user_movie_review, movie_user_review, movie):
    total = 0
    i = 0
    mean = 0
    for user_reviews in movie_user_review.keys():
        if user_reviews == movie:
            for rater in user_movie_review.keys():
                if rater in movie_user_review[movie].keys():
                    i += 1
                    total += int(movie_user_review[movie][rater])
        if i != 0:
            mean = round(total / i)
    return mean


def avg(user_movie_review, movie_user_review, test_user_movie_review):
    predicted_user_movie_reviews = {}
    difference = []
    for user in test_user_movie_review.keys():
        predicted_user_movie_reviews[user] = {}
        for movie in test_user_movie_review[user].keys():
            # print(predicted_user_movie_reviews[user])
            if (movie not in user_movie_review[user].keys()):
                if (user not in predicted_user_movie_reviews.keys()):
                    mean_all = mean_find(user_movie_review, movie_user_review,
                                         movie)
                    predicted_user_movie_reviews[user] = {movie: mean_all}
                # round(mean(int(rating) for user_reviews in
                # movie_user_review[movie] for rating in user_reviews))}
                else:
                    mean_all = mean_find(user_movie_review, movie_user_review,
                                         movie)
                    predicted_user_movie_reviews[user].update({movie: mean_all}
                                                              )
                    # round(mean(int(rating) for user_reviews in
                    # movie_user_review[movie] for rating in user_reviews))})
            difference.append(abs(
                int(test_user_movie_review[user][movie]) - int(
                    predicted_user_movie_reviews[user][movie])))
    result = mean(difference)
    return result


def rating_ed_diff_gen(intersection_watched_list, curr_user, user,
                       movie_user_review):
    distance = 0
    for movie in intersection_watched_list:
        distance += abs(int(movie_user_review[movie][user]) - int(
            movie_user_review[movie][curr_user])) ** 2
    return distance ** 0.5


def rating_md_diff_gen(intersection_watched_list, curr_user, user,
                       movie_user_review):
    distance = 0
    for movie in intersection_watched_list:
        distance += abs(int(movie_user_review[movie][user]) - int(
            movie_user_review[movie][curr_user]))
    return distance


def rating_lmax_diff_gen(intersection_watched_list, curr_user, user,
                         movie_user_review):
    distance = []
    for movie in intersection_watched_list:
        distance.append(abs(int(movie_user_review[movie][user]) - int(
            movie_user_review[movie][curr_user])))
    return max(distance)


def top_pick_rating(sorted_rating_map, test_movie, movie_user_review, i):
    top_user_rating_sum = 0
    j = 0
    for pick in sorted_rating_map:
        j += 1
        top_user_rating_sum += int(movie_user_review[test_movie][pick[0]])
        return round(top_user_rating_sum / j)


def estimate(test_user, test_movie, user_movie_review, movie_user_review,
             predictor_type, i):
    test_user_watch_list = user_movie_review[test_user].keys()
    rating_difference_map = []
    for curr_user in movie_user_review[test_movie].keys():
        rating_difference = 0
        curr_user_watch_list = user_movie_review[curr_user].keys()
        intersection_watch_list = set(test_user_watch_list).intersection(
            set(curr_user_watch_list))
        if len(intersection_watch_list) > 10:
            if predictor_type == "ED":
                rating_difference = rating_ed_diff_gen(intersection_watch_list,
                                                       test_user, curr_user,
                                                       movie_user_review)
            elif predictor_type == "MD":
                rating_difference = rating_md_diff_gen(intersection_watch_list,
                                                       test_user, curr_user,
                                                       movie_user_review)
            elif predictor_type == "Lmax":
                rating_difference = rating_lmax_diff_gen(
                    intersection_watch_list, test_user, curr_user,
                    movie_user_review)
            rating_difference_map.append(
                [curr_user, rating_difference / len(intersection_watch_list)])
        else:
            rating_difference_map.append([curr_user,
                                          mean_find(user_movie_review,
                                                    movie_user_review,
                                                    test_movie)])
    sorted_rating_diff_map = \
        sorted(rating_difference_map, key=itemgetter(1))[:i]
    # k Value in question is in the square braces here
    # print(sorted_rating_diff_map)
    return top_pick_rating(sorted_rating_diff_map, test_movie,
                           movie_user_review, i)
    # return mean_find(user_movie_review, movie_user_review, test_movie)


def ed_md_lmax(user_movie_review, movie_user_review, test_user_movie_review,
               predictor_type, i):
    predicted_user_movie_reviews = {}
    for user in list(test_user_movie_review.keys()):  # [:1]:
        # print(user)
        predicted_user_movie_reviews[user] = {}
        # print(test_movie_user_review[user].keys())
        difference = 0
        j = 0
        for movie in test_user_movie_review[user].keys():
            if movie in movie_user_review.keys():
                predicted_user_movie_reviews[user][movie] = \
                    estimate(user, movie, user_movie_review,
                             movie_user_review, predictor_type, i)
                # difference += (abs(int(test_user_movie_review[user][movie]) -
                #                     int(predicted_user_movie_reviews[user]
                #                         [movie])))
    # print(predicted_user_movie_reviews)
    for user in predicted_user_movie_reviews.keys():
        for movie in predicted_user_movie_reviews[user].keys():
            j += 1
            difference += (abs(int(test_user_movie_review[user][movie]) - int(
                predicted_user_movie_reviews[user][movie])))
    result = difference / j
    return result


def predictor(user_movie_review, movie_user_review, test_user_movie_review,
              predictor_type, i):
    if predictor_type == "AVG":
        result = avg(user_movie_review, movie_user_review,
                     test_user_movie_review)
        return result
    else:
        result = ed_md_lmax(user_movie_review, movie_user_review,
                            test_user_movie_review, predictor_type, i)
        return result


user_movie_review, movie_user_review = generate_user_data(
    "u1.base")  # input("File name:"))
test_user_movie_review, test_movie_user_review = generate_user_data("u1.test")

user_profile = generate_user_profile()

for i in [5, 7, 10]:
    print("Printing for key = ", i)
    result = predictor(user_movie_review, movie_user_review,
                       test_user_movie_review, "ED", i)
    print("Eculidean Distance   :", result)
    result = predictor(user_movie_review, movie_user_review,
                       test_user_movie_review, "MD", i)
    print("Manhattan Distance   :", result)
    result = predictor(user_movie_review, movie_user_review,
                       test_user_movie_review, "Lmax", i)
    print("Lmax Distance        :", result)
