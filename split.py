with open("file.txt", 'w') as output:
    for row in values:
        output.write(str(row) + '\n')