n = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 
'nine', 'ten', 'over', 'under']
for i in range(13):
    for j in range(13):
        if i == 11:
            print('{},{},{}'.format(n[i], n[j], n[i]))
        elif j == 11:
            print('{},{},{}'.format(n[i], n[j], n[12]))
        elif i == 12:
            print('{},{},{}'.format(n[i], n[j], n[i]))
        elif j == 12:
            print('{},{},{}'.format(n[i], n[j], n[11]))
        else:
            print('{},{},{}'.format(
                n[i],
                n[j],
                n[i-j] if 0 <= i-j <= 10 else (
                    'over' if i-j > 10 else 'under')))
