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
    print("generate_dict")
    sample = {}
    for list in line_split:
        if list[placeholder1] not in sample.keys():
            sample[list[placeholder1]] = {list[placeholder2]: list[2]}
        else:
            sample[list[placeholder1]][list[placeholder2]] = list[2]
    return sample


def matched_users_gen(curr_user, user_profile, movie, user_movie_reviews, key):
    matched_users = []
    for u_num in user_profile.keys():
        u_num_watched_list = [movie for movie in user_movie_review[u_num]]
        if u_num != curr_user and movie in u_num_watched_list:
            # print(user_profile)
            if (abs(int(user_profile[u_num][0]) - int(
                    user_profile[curr_user][0])) < 5) and (
                        user_profile[u_num][1] == user_profile[curr_user][1]) \
                    and (user_profile[u_num][2] == user_profile[curr_user][2])\
                    and key == 0:
                matched_users.append(u_num)
            elif (abs(int(user_profile[u_num][0]) - int(
                    user_profile[curr_user][0])) < 5) and (
                        user_profile[u_num][2] == user_profile[curr_user][2]) \
                    and key == 1:
                matched_users.append(u_num)
            elif (abs(int(user_profile[u_num][0]) - int(
                    user_profile[curr_user][0])) < 5) and (
                        user_profile[u_num][1] == user_profile[curr_user][1]) \
                    and key == 2:
                matched_users.append(u_num)
            elif (user_profile[u_num][2] == user_profile[curr_user][2]) \
                    and key == 3:
                matched_users.append(u_num)
            elif (abs(int(user_profile[u_num][0]) - int(
                    user_profile[curr_user][0])) < 10) and key == 4:
                matched_users.append(u_num)
            elif (abs(int(user_profile[u_num][0]) - int(
                    user_profile[curr_user][0])) < 20) and key == 5:
                matched_users.append(u_num)
    return matched_users


def rating_diff_gen(intersection_watched_list, curr_user, user,
                    movie_user_review):
    distance = 0
    for movie in intersection_watched_list:
        distance += abs(int(movie_user_review[movie][user]) - int(
            movie_user_review[movie][curr_user]))
    return distance / len(intersection_watched_list)


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


def top_pick_rating(sorted_rating_map, curr_moive, movie_user_review):
    top_user_rating_sum = 0
    i = 0
    # print(len(sorted_euclid_map))
    for pick in sorted_rating_map:
        i += 1
        top_user_rating_sum += int(movie_user_review[curr_moive][pick[0]])
        if i == 10:
            return round(top_user_rating_sum / 10)
    return round(top_user_rating_sum / i)


def estimate(curr_user: object, curr_movie: object, user_movie_review: object,
             movie_user_review: object,
             matched_users_list: object, predictor_type: object) -> object:
    rating_difference_map = []
    curr_user_watch_list = user_movie_review[curr_user].keys()
    for user in matched_users_list:
        if user in movie_user_review[curr_movie]:
            user_watch_list = user_movie_review[user].keys()
            intersection_watch_list = list(
                set(curr_user_watch_list).intersection(set(user_watch_list)))
            rating_difference = 0
            if len(intersection_watch_list) > 0:
                if predictor_type == "ED":
                    rating_difference = rating_ed_diff_gen(
                        intersection_watch_list, curr_user, user,
                        movie_user_review)
                elif predictor_type == "MD":
                    rating_difference = rating_ed_diff_gen(
                        intersection_watch_list, curr_user, user,
                        movie_user_review)
                elif predictor_type == "Lmax":
                    rating_difference = rating_ed_diff_gen(
                        intersection_watch_list, curr_user, user,
                        movie_user_review)
            else:
                return False
            if user in movie_user_review[curr_movie]:
                # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                # print(rating_difference)
                # checking if the user has rated the current movie
                rating_difference_map.append([user, rating_difference])
    if rating_difference_map != []:
        sorted_rating_diff_map = sorted(rating_difference_map,
                                        key=itemgetter(1))[:5]
        # print(sorted_euclid_map)
        return round(top_pick_rating(sorted_rating_diff_map, curr_movie,
                                     movie_user_review))
    else:
        return False


def predictor(user_movie_review, movie_user_review, test_user_movie_review,
              user_profile, predictor_type):
    print("predictor")
    difference = 0
    predicted_user_movie_reviews = {}
    for user in test_user_movie_review.keys():
        predicted_user_movie_reviews[user] = {}
        for movie in test_movie_user_review.keys():
            # print(predicted_user_movie_reviews[user])
            matched_users_list = []
            if (movie not in user_movie_review[user].keys()) and \
                    (movie not in predicted_user_movie_reviews[user].keys()):
                # print(curr_user_watch_list)
                for i in range(6):
                    # print(user)
                    matched_users_list = matched_users_gen(user, user_profile,
                                                           movie,
                                                           movie_user_review,
                                                           i)
                    if len(matched_users_list) > 0:
                        temp = estimate(user, movie, user_movie_review,
                                        movie_user_review, matched_users_list,
                                        predictor_type)
                        if user not in predicted_user_movie_reviews.keys() \
                                and temp != False:
                            predicted_user_movie_reviews[user] = {movie: temp}
                        elif temp != False:
                            predicted_user_movie_reviews[user].update(
                                {movie: temp})
                        elif i == 5 and temp == False and user not in\
                                predicted_user_movie_reviews.keys():
                            predicted_user_movie_reviews[user] = {movie: round(
                                mean(int(rating) for user_reviews in
                                     movie_user_review[movie] for rating in
                                     user_reviews))}
                        elif i == 5 and temp == False and user in \
                                predicted_user_movie_reviews.keys():
                            predicted_user_movie_reviews[user].update({
                                movie: round(mean(int(rating)
                                                  for user_reviews in
                                                  movie_user_review[movie]
                                                  for rating in user_reviews))}
                            )

    for user in predicted_user_movie_reviews.keys():
        for movie in predicted_user_movie_reviews[user].keys():
            if movie in movie_user_review.keys():
                i += 1
                difference += (abs(
                    int(test_user_movie_review[user][movie]) - int(
                        predicted_user_movie_reviews[user][movie])))
    result = difference / i
    return result


user_movie_review, movie_user_review = generate_user_data("u1.base")
test_user_movie_review, test_movie_user_review = generate_user_data("u1.test")
user_profile = generate_user_profile()

print("Printing for key = 10")
result = predictor(user_movie_review, movie_user_review,
                   test_user_movie_review, user_profile, "ED")
print("Eculidean Distance   :", result)
result = predictor(user_movie_review, movie_user_review,
                   test_user_movie_review, user_profile, "MD")
print("Manhattan Distance   :", result)
result = predictor(user_movie_review, movie_user_review,
                   test_user_movie_review, user_profile, "Lmax")
print("Lmax Distance        :", result)
