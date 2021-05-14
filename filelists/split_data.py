from path import Path



temp = Path('/home/huimingsun/Desktop/wav2lip/videos/lrs2_preprocessed/HQ_Images').listdir()
temp = [str(i.name) for i in temp]
with open("train.txt", 'w') as  output:
    for row in temp[:int(len(temp)*0.9)]:
        output.write(str(row) + '\n')


with open("val.txt", 'w') as output:
    for row in temp[int(len(temp)*0.9):]:
        output.write(str(row) + '\n')