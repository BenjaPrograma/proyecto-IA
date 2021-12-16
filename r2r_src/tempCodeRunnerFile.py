def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

print(intersection("the bathroom door".split(' '), 'they suck'.split(' ')))