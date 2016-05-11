#This is code to detect the number of positive and negative words in the file given. Here I have taken a tweet file and given it as input. The program counts the number of positive and negative words in each tweet and decides the type of tweet.

def read_file(file_name):
    #creating positive word list
    file = open(file_name, "r",encoding='UTF-8')
    file_lines = file.readlines()
    file_lines_word = []
    for line in file_lines:
        file_lines_word.append(line[:-1])
    return file_lines_word

def tweet_counter(tweets_list,  pos_word_list, neg_word_list):
    tot_neg_tweets = 0
    tot_pos_tweets = 0
    tot_mixed_tweets = 0
    neg_word_count = 0
    pos_word_count = 0
    for tweet in tweets_list:
        pos_count_tweet = 0
        neg_count_tweet = 0
        for word in tweet:
            if word in pos_word_list or word == (":)" or ":-)" or ":-D" or "<3" or "♥"):
                pos_count_tweet += 1
                pos_word_count += 1
            elif word in neg_word_list or word == (":(" or ":-("):
                neg_count_tweet += 1
                neg_word_count += 1
            else:
                pass
        weight_tweet = neg_count_tweet - pos_count_tweet
        if weight_tweet < 0:
           tot_neg_tweets += 1
        elif weight_tweet > 0:
            tot_pos_tweets += 1
        else:
            tot_mixed_tweets += 1
    return tot_pos_tweets,tot_neg_tweets,tot_mixed_tweets,neg_word_count,pos_word_count

def hw1_start():
    pos_word_list = read_file("pos.wn")
    neg_word_list = read_file("neg.wn")
    tweets_list = [word.split() for word in read_file(input("Enter your full file name with extension"))]
    for tweets in tweets_list:
        print(tweets)
    tot_pos_tweets,tot_neg_tweets,tot_mixed_tweets,neg_word_count,pos_word_count = tweet_counter(tweets_list, pos_word_list, neg_word_list)
    print("Total number of positive tweets:",tot_pos_tweets)
    print("Total number of negative tweets:",tot_neg_tweets)
    print("Total number of mixed tweets:",tot_mixed_tweets)
    print("Total number of negative words:",neg_word_count)
    print("Total number of positive words:",pos_word_count)
hw1_start()