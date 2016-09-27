# from operator import itemgetter
# from statistics import mean

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


def predictor(user_movie_review, movie_user_review):
    predicted_user_movie_reviews = {}
    for user in user_movie_review.keys():
        predicted_user_movie_reviews[user] = {}
        for movie in movie_user_review.keys():
            # print(predicted_user_movie_reviews[user])
            if (movie not in user_movie_review[user].keys()):
                if (user not in predicted_user_movie_reviews.keys()):
                    mean = mean_find(user_movie_review, movie_user_review,
                                     movie)
                    predicted_user_movie_reviews[user] = {movie: mean}
                # round(mean(int(rating) for user_reviews in
                # movie_user_review[movie] for rating in user_reviews))}
                else:
                    mean = mean_find(user_movie_review, movie_user_review,
                                     movie)
                    predicted_user_movie_reviews[user].update({movie: mean})
                    # round(mean(int(rating) for user_reviews in
                    # movie_user_review[movie] for rating in user_reviews))})
    return predicted_user_movie_reviews


user_movie_review, movie_user_review = generate_user_data(
    "u1.base")  # input("File name:"))

user_profile = generate_user_profile()
predicted_user_movie_reviews1 = predictor(user_movie_review, movie_user_review)
print("result")
print(predicted_user_movie_reviews1)
file = open("u1_base_avg_review.DATA File", "w")
for user in predicted_user_movie_reviews1.keys():
    for movie in predicted_user_movie_reviews1[user].keys():
        file.write(user + "   " + movie + "  " + str(
            predicted_user_movie_reviews1[user][movie]) + "\n")
