price_per_length = {1:1,2:4,3:3,4:2,5:2,6:5,7:17,8:20,9:24,10:30}

given_rod_length = 7

price_of_opt_cut_rod_of_len_k = {0:0}
cut_per_length = {}

rod_length = 1
price_of_opt_cut_rod_of_len_k[rod_length] = price_per_length[rod_length]

while(rod_length<given_rod_length):
    rod_length += 1
    cut_value = []
    for i in range(1,rod_length+1):
        if i == rod_length:
            cut_value.append(price_per_length[i]+price_of_opt_cut_rod_of_len_k[rod_length-i])
        else:
            cut_value.append(price_per_length[i] + price_of_opt_cut_rod_of_len_k[rod_length - i])
    cut_per_length[rod_length] = rod_length - (cut_value.index(max(cut_value)) + 1)
    price_of_opt_cut_rod_of_len_k[rod_length] = max(cut_value)

print(price_of_opt_cut_rod_of_len_k)
print(cut_per_length)