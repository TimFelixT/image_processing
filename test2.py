def format_number(n):
    string2 = str(n)
    string2.replace('00', '23')
    for i in string2:
        string2 += i + ','
    return string2


if __name__ == '__main__':
    print(format_number(1000000))

