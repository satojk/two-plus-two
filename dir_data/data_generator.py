n = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
     'nine', 'ten', 'over', 'under']
for i in range(13):
    for j in range(13):
        if i == 11:
            print('{},{},{},{}'.format(n[i], n[j], n[i], n[i]))
        elif j == 11:
            print('{},{},{},{}'.format(n[i], n[j], n[j], n[j]))
        elif i == 12:
            print('{},{},{},{}'.format(n[i], n[j], n[i], n[i]))
        elif j == 12:
            print('{},{},{},{}'.format(n[i], n[j], n[j], n[j]))
        else:
            fst_dig = (i + j) // 10
            snd_dig = (i + j) % 10
            print('{},{},{},{}'.format(
                n[i],
                n[j],
                n[fst_dig],
                n[snd_dig]))
